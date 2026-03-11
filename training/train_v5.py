"""
V5 Training: Verifiable Milestone Hybrid Reward
Reward = 0.7 * final_binary + 0.3 * (verified_milestones / total_milestones)

Key differences from other variants:
- v2a (outcome-dominant hybrid): failure → steps_completed/steps_total (gap ~0.2)
- v2c (naive stepwise partial): per-step format credit, no API verification
- v5 (verifiable milestone hybrid): 0.7*binary + 0.3*milestones (gap ~0.76)

Design principles:
1. Milestones = verified API state transitions (not format parsing)
2. Sequential gating (verifier breaks on first error)
3. Final binary dominates (0.7 vs 0.3)
4. Uniform milestone weights (no hand-tuned per-step values)

Usage:
    cd /tmp/rlvr_gym
    nohup /usr/bin/python3 /tmp/train_v5.py > /tmp/train_v5.log 2>&1 &
"""
import json
import subprocess
import time
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

SCRIPT_DIR = "/tmp/rlvr_gym"
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from datasets import Dataset

# ===== Config (MUST match v2a exactly for clean ablation) =====
# v2a actual config verified from train_launcher.py + model config.json:
#   hidden_size=896, num_hidden_layers=24 → Qwen2.5-0.5B-Instruct
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512  # match train_grpo.py default
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-6
BETA = 0.04
NUM_TRAIN_EPOCHS = 3
TOTAL_TASKS = 100
API_PORT = 8000

# Import SYSTEM_PROMPT from existing training module
from train_grpo import SYSTEM_PROMPT, build_dataset, format_reward


def v5_subgoal_reward(completions, expected_actions=None, instruction=None,
                      task_type=None, difficulty=None, **kwargs):
    """Verifiable milestone hybrid reward.

    Reward = 0.7 * final_binary + 0.3 * (verified_milestones / total_milestones)

    Milestones are counted by the verifier's steps_completed, which requires:
    - Correct HTTP method
    - Correct path pattern
    - Successful API execution (200 response = state transition happened)
    - Sequential gating (verifier breaks on first error)

    FIX vs v2a's accuracy_reward:
    - Passes REAL task_type (not "unknown") so final state verification works
    - v2a had task_type="unknown" → _get_expected_final_state() returns None → skips check

    Key difference from v2a: v5's gap between "almost done" and "done" is 0.76,
    vs v2a's gap of ~0.2. This makes partial credit harvesting far less attractive.
    """
    from task_generator import Task
    from verifier import execute_and_verify

    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)

        try:
            exp_actions = json.loads(expected_actions[i]) if expected_actions else []
            # FIX: pass real task_type so verifier can check final state
            real_task_type = task_type[i] if task_type else "unknown"
            real_difficulty = int(difficulty[i]) if difficulty else 1
            task = Task(
                instruction=instruction[i] if instruction else "",
                expected_actions=exp_actions,
                difficulty=real_difficulty,
                task_type=real_task_type,
            )

            result = execute_and_verify(task, text)

            # Binary component: 1.0 only if final state matches expected
            binary = 1.0 if result.success else 0.0

            # Milestone component: fraction of verified state transitions
            # steps_completed requires: method match + path match + API 200
            # Sequential gating: verifier breaks on first error
            steps_total = result.steps_total
            milestone_fraction = result.steps_completed / steps_total if steps_total > 0 else 0.0

            # Combined reward: final binary dominates (0.7 vs 0.3)
            reward = 0.7 * binary + 0.3 * milestone_fraction

            if i == 0:
                print(f"[V5] success={result.success} steps={result.steps_completed}/{steps_total} "
                      f"binary={binary:.1f} milestone={milestone_fraction:.2f} reward={reward:.3f}")

            rewards.append(reward)

        except Exception as e:
            if i == 0:
                print(f"[V5] EXCEPTION: {e}")
            rewards.append(0.0)

    return rewards


def start_api_server():
    """Start the FastAPI server in background."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_system:app",
         "--port", str(API_PORT), "--host", "127.0.0.1",
         "--log-level", "warning"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    print(f"[*] API server started on port {API_PORT} (PID: {proc.pid})")
    return proc


def main():
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer

    print("=" * 60)
    print("  V5: Verifiable Milestone Hybrid Reward")
    print("  Reward = 0.7 * final_binary + 0.3 * milestones")
    print("  From scratch (Qwen2.5-0.5B-Instruct)")
    print("=" * 60)

    # Start API server
    api_proc = start_api_server()

    try:
        # Build dataset
        print(f"\n[*] Generating {TOTAL_TASKS} training tasks...")
        dataset = build_dataset(TOTAL_TASKS, start_difficulty=1)
        print(f"[OK] Dataset: {len(dataset)} tasks")

        d_counts = {}
        for d in dataset["difficulty"]:
            d_counts[d] = d_counts.get(d, 0) + 1
        print(f"     Difficulty: {d_counts}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # GRPO Config
        training_args = GRPOConfig(
            output_dir="/tmp/rlvr_gym/checkpoints_v5",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            beta=BETA,
            logging_steps=5,
            save_steps=50,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            temperature=0.9,
        )

        # Initialize trainer with v5 reward
        print(f"[*] Initializing GRPOTrainer...")
        print(f"    Model: {MODEL_NAME}")
        print(f"    Reward: format_reward + v5_subgoal_reward")
        print(f"    Group size: {NUM_GENERATIONS}")
        print(f"    KL beta: {BETA}")
        print(f"    Epochs: {NUM_TRAIN_EPOCHS}")

        trainer = GRPOTrainer(
            model=MODEL_NAME,
            reward_funcs=[format_reward, v5_subgoal_reward],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Train
        print(f"\n[*] Starting V5 GRPO training...")
        trainer.train()

        # Save
        output_path = "/tmp/rlvr_gym/final_model_v5"
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"\n[OK] V5 model saved to {output_path}")

    finally:
        api_proc.terminate()
        api_proc.wait()
        print("[*] API server stopped")


if __name__ == "__main__":
    main()
