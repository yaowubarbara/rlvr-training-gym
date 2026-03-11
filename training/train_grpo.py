"""
API Training Gym - Module 4: GRPO Training Loop
Uses TRL's GRPOTrainer with verifiable rewards (no learned reward model).

Architecture reference:
- TRL GRPOTrainer for training loop
- Reasoning Gym style for task generation
- NeMo Gym pattern for API verification

Requirements:
    pip install trl transformers datasets accelerate
    # API server must be running: uvicorn api_system:app --port 8000
"""
import json
import subprocess
import time
import signal
import os
import sys
from datasets import Dataset

# ===== Config =====
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_GENERATIONS = 8          # Group size for GRPO (z-score normalization)
MAX_COMPLETION_LENGTH = 512  # Max tokens per completion
BATCH_SIZE = 2               # Per-device batch size (small for 8GB VRAM)
GRADIENT_ACCUMULATION = 4    # Effective batch = 2 * 4 = 8
LEARNING_RATE = 1e-6         # Conservative for online RL
BETA = 0.04                  # KL penalty coefficient
NUM_TRAIN_EPOCHS = 3
TOTAL_TASKS = 200            # Training tasks per epoch
CURRICULUM_START = 1         # Start difficulty
API_PORT = 8000

# ===== System prompt for the agent =====
SYSTEM_PROMPT = """You are an API agent. You interact with an Order Management API by outputting structured API calls.

## API Endpoints:
1. POST /orders — Create order (body: {"customer_name": str, "product": str, "quantity": int, "price": float})
2. GET /orders/{id} — Get order details
3. POST /orders/{id}/pay — Pay (body: {"order_id": str, "amount": float, "method": "card"|"bank_transfer"})
4. POST /orders/{id}/ship — Ship (body: {"order_id": str, "address": str, "carrier": "dhl"|"fedex"|"ups"})
5. POST /orders/{id}/deliver — Mark delivered (no body needed)
6. POST /orders/{id}/cancel — Cancel (only created/paid orders, no body)
7. POST /orders/{id}/refund — Refund (body: {"order_id": str, "reason": str}, only paid/delivered)

## State Machine:
created → paid → shipped → delivered
created → cancelled
paid → cancelled | refunded
delivered → refunded

## Rules:
- Payment amount must EXACTLY equal quantity × price
- order_id in body must match the URL parameter
- Carrier: dhl, fedex, ups. Payment method: card, bank_transfer

## Output Format:
You MUST output actions in this EXACT format (one per line):

ACTION: METHOD /path
BODY: {"key": "value"}

Example:
ACTION: POST /orders
BODY: {"customer_name": "Alice", "product": "Laptop", "quantity": 1, "price": 999.99}

ACTION: POST /orders/abc123/pay
BODY: {"order_id": "abc123", "amount": 999.99, "method": "card"}

IMPORTANT: Replace the order_id with the actual ID from the create response. Use the EXACT total (quantity × price) for payment amount."""


def build_dataset(n_tasks: int, start_difficulty: int = 1) -> Dataset:
    """Generate training dataset from task generator with curriculum."""
    # Import here to avoid circular dependency issues
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task_generator import generate_curriculum

    tasks = generate_curriculum(n_tasks, start_difficulty)

    prompts = []
    metadata = []

    for task in tasks:
        # Build the user message
        user_msg = task.instruction

        # Format as chat messages for the model
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(prompt)
        metadata.append({
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "expected_actions": json.dumps(task.expected_actions),
            "instruction": task.instruction,
        })

    return Dataset.from_dict({
        "prompt": prompts,
        "task_type": [m["task_type"] for m in metadata],
        "difficulty": [m["difficulty"] for m in metadata],
        "expected_actions": [m["expected_actions"] for m in metadata],
        "instruction": [m["instruction"] for m in metadata],
    })


# ===== Reward Functions (TRL composable pattern) =====

def format_reward(completions, **kwargs):
    """Check if the agent's output follows the correct ACTION/BODY format.

    Returns 1.0 if output contains at least one valid ACTION line.
    Returns 0.0 if output is malformed.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        has_action = "ACTION:" in text
        has_method = any(m in text for m in ["POST", "GET", "PUT", "DELETE"])
        rewards.append(1.0 if (has_action and has_method) else 0.0)
    return rewards


def accuracy_reward(completions, expected_actions=None, instruction=None, **kwargs):
    """Execute agent's actions against the API and verify correctness.

    This is the core RLVR reward — deterministic, verifiable, no learned model.
    Returns 1.0 if all actions succeed and final state is correct.
    Returns partial credit (0.0-1.0) based on steps completed.
    """
    import httpx
    from task_generator import Task

    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)

        try:
            # Reconstruct task for verification
            exp_actions = json.loads(expected_actions[i]) if expected_actions else []
            task = Task(
                instruction=instruction[i] if instruction else "",
                expected_actions=exp_actions,
                difficulty=1,
                task_type="unknown",
            )

            # Use verifier
            from verifier import execute_and_verify
            result = execute_and_verify(task, text)

            # Debug: log first generation of each batch
            if i == 0:
                print(f"[DEBUG] text[:200]: {repr(text[:200])}")
                print(f"[DEBUG] success={result.success} partial={result.partial_reward} "
                      f"steps={result.steps_completed}/{result.steps_total} "
                      f"errors={result.errors[:2]}")

            # Use partial reward to give gradient signal even on failures
            # Full credit for success, partial for getting some steps right
            rewards.append(result.partial_reward if not result.success else 1.0)

        except Exception as e:
            if i == 0:
                print(f"[DEBUG] EXCEPTION in accuracy_reward: {e}")
            rewards.append(0.0)

    return rewards


def start_api_server():
    """Start the FastAPI server in background."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_system:app",
         "--port", str(API_PORT), "--host", "127.0.0.1",
         "--log-level", "warning"],
        cwd=script_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # Wait for server to start
    print(f"[*] API server started on port {API_PORT} (PID: {proc.pid})")
    return proc


def main():
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60)
    print("  API Training Gym — GRPO Training")
    print("  Architecture: NeMo Gym + Reasoning Gym + TRL")
    print("=" * 60)

    # Start API server
    api_proc = start_api_server()

    try:
        # Build dataset with curriculum
        print(f"\n[*] Generating {TOTAL_TASKS} training tasks (curriculum: {CURRICULUM_START}→3)...")
        dataset = build_dataset(TOTAL_TASKS, CURRICULUM_START)
        print(f"[OK] Dataset ready: {len(dataset)} tasks")
        print(f"     Difficulty distribution: "
              f"easy={sum(1 for d in dataset['difficulty'] if d==1)}, "
              f"medium={sum(1 for d in dataset['difficulty'] if d==2)}, "
              f"hard={sum(1 for d in dataset['difficulty'] if d==3)}")

        # Load model
        print(f"\n[*] Loading model: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # GRPO Config (TRL pattern)
        training_args = GRPOConfig(
            output_dir="/tmp/rlvr_gym/checkpoints",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            # GRPO-specific
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            beta=BETA,
            # Logging
            logging_steps=5,
            save_steps=50,
            report_to="none",  # or "wandb" if you have it
            # Performance
            bf16=True,
            gradient_checkpointing=True,
            # Temperature for generation diversity
            temperature=0.9,
        )

        # Initialize GRPO Trainer with composable rewards (TRL pattern)
        print(f"[*] Initializing GRPOTrainer...")
        print(f"    Group size: {NUM_GENERATIONS}")
        print(f"    KL beta: {BETA}")
        print(f"    Learning rate: {LEARNING_RATE}")

        trainer = GRPOTrainer(
            model=MODEL_NAME,
            reward_funcs=[format_reward, accuracy_reward],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Train!
        print(f"\n[*] Starting GRPO training...")
        print(f"    Total tasks: {TOTAL_TASKS}")
        print(f"    Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
        print(f"    Epochs: {NUM_TRAIN_EPOCHS}")
        print(f"    Steps per epoch: ~{TOTAL_TASKS // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")

        trainer.train()

        # Save final model
        output_path = "/tmp/rlvr_gym/final_model"
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"\n[OK] Model saved to {output_path}")

        # Quick evaluation
        print("\n[*] Running quick eval on 20 test tasks...")
        eval_dataset = build_dataset(20, start_difficulty=1)
        correct = 0
        total = 0
        for i in range(min(20, len(eval_dataset))):
            prompt = eval_dataset[i]["prompt"]
            # Generate
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
            outputs = trainer.model.generate(
                inputs.to(trainer.model.device),
                max_new_tokens=MAX_COMPLETION_LENGTH,
                temperature=0.1,  # Low temp for eval
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

            # Verify
            exp = json.loads(eval_dataset[i]["expected_actions"])
            task = __import__("task_generator").Task(
                instruction=eval_dataset[i]["instruction"],
                expected_actions=exp,
                difficulty=eval_dataset[i]["difficulty"],
                task_type=eval_dataset[i]["task_type"],
            )
            from verifier import execute_and_verify
            result = execute_and_verify(task, response)
            if result.success:
                correct += 1
            total += 1
            print(f"  Task {i+1}: {'PASS' if result.success else 'FAIL'} "
                  f"({result.steps_completed}/{result.steps_total} steps)")

        print(f"\n[RESULT] Eval accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    finally:
        # Cleanup
        api_proc.terminate()
        api_proc.wait()
        print("[*] API server stopped")


if __name__ == "__main__":
    main()
