#!/usr/bin/env python3
"""v2b: Dynamic Boltzmann Curriculum — regenerate tasks each epoch based on reward.
Hypothesis: static dataset wastes compute on mastered tasks.
Fix: use BoltzmannCurriculum to shift probability toward harder tasks as model improves.
"""
import subprocess, sys, os, time, json

SCRIPT_DIR = "/tmp/rlvr_gym"
API_PORT = 8000
PYTHON = "/usr/bin/python3"

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def start_api():
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "api_system:app", "--port", str(API_PORT),
         "--host", "127.0.0.1", "--log-level", "warning"],
        cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    print(f"[*] API server started (PID: {proc.pid})")
    return proc

def run_training():
    sys.path.insert(0, SCRIPT_DIR)
    os.chdir(SCRIPT_DIR)

    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer
    from train_grpo import format_reward, accuracy_reward, SYSTEM_PROMPT
    from task_generator import generate_task, BoltzmannCurriculum
    from datasets import Dataset

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    NUM_GENERATIONS = 4
    MAX_COMPLETION = 384
    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    TASKS_PER_EPOCH = 100
    NUM_EPOCHS = 3

    print("=" * 60)
    print("  RLVR Training Gym — v2b: Dynamic Boltzmann Curriculum")
    print("=" * 60)

    print(f"\n[*] Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === KEY CHANGE: Boltzmann curriculum with per-epoch regeneration ===
    curriculum = BoltzmannCurriculum(
        tau_start=2.0,
        tau_end=0.3,
        difficulties=[1, 2, 3],
    )

    all_rewards_by_epoch = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Generate tasks using current curriculum distribution
        progress = epoch / NUM_EPOCHS  # 0.0 → 0.67 over 3 epochs
        probs = curriculum.get_difficulty_probs(progress)
        tau = curriculum.get_tau(progress)
        print(f"[*] Curriculum probs: easy={probs[1]:.2f}, med={probs[2]:.2f}, hard={probs[3]:.2f}")
        print(f"    Temperature (tau): {tau:.3f}")

        tasks = []
        for _ in range(TASKS_PER_EPOCH):
            diff = curriculum.sample_difficulty(progress)
            tasks.append(generate_task(diff))

        prompts = []
        metadata = []
        for task in tasks:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ]
            prompts.append(prompt)
            metadata.append({
                "task_type": task.task_type,
                "difficulty": task.difficulty,
                "expected_actions": json.dumps(task.expected_actions),
                "instruction": task.instruction,
            })

        dataset = Dataset.from_dict({
            "prompt": prompts,
            "task_type": [m["task_type"] for m in metadata],
            "difficulty": [m["difficulty"] for m in metadata],
            "expected_actions": [m["expected_actions"] for m in metadata],
            "instruction": [m["instruction"] for m in metadata],
        })

        dist = {1: 0, 2: 0, 3: 0}
        for d in dataset['difficulty']:
            dist[d] += 1
        print(f"[OK] Dataset: {len(dataset)} tasks (easy={dist[1]}, med={dist[2]}, hard={dist[3]})")

        config = GRPOConfig(
            output_dir=f"/tmp/rlvr_gym/checkpoints_v2b_epoch{epoch+1}",
            num_train_epochs=1,  # 1 epoch per regeneration cycle
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=1e-6,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION,
            beta=0.04,
            logging_steps=1,
            save_steps=100,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            temperature=0.9,
            max_grad_norm=1.0,
            per_device_eval_batch_size=1,
        )

        if epoch == 0:
            trainer = GRPOTrainer(
                model=MODEL,
                reward_funcs=[format_reward, accuracy_reward],
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
        else:
            # Update dataset for existing trainer
            trainer.args = config
            trainer.train_dataset = dataset

        print(f"[*] Training epoch {epoch+1}...")
        trainer.train()

        # Collect rewards from this epoch to update curriculum
        # Parse the training log for this epoch's rewards
        epoch_rewards = {1: [], 2: [], 3: []}
        # Use a quick eval to estimate per-difficulty reward
        from verifier import execute_and_verify
        print(f"\n[*] Evaluating per-difficulty reward...")
        for diff in [1, 2, 3]:
            eval_tasks = [generate_task(diff) for _ in range(5)]
            for etask in eval_tasks:
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": etask.instruction},
                ]
                inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
                outputs = trainer.model.generate(
                    inputs.to(trainer.model.device),
                    max_new_tokens=384, temperature=0.1, do_sample=True,
                )
                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                result = execute_and_verify(etask, response)
                reward = 1.0 if result.success else result.partial_reward
                epoch_rewards[diff].append(reward)

        # Update curriculum with observed rewards
        for diff in [1, 2, 3]:
            avg_reward = sum(epoch_rewards[diff]) / len(epoch_rewards[diff]) if epoch_rewards[diff] else 0
            curriculum.update_reward(diff, avg_reward)
            print(f"  Difficulty {diff}: avg_reward={avg_reward:.3f}")

        all_rewards_by_epoch.append(epoch_rewards)
        next_progress = (epoch + 1) / NUM_EPOCHS
        print(f"[*] Curriculum updated. Next tau: {curriculum.get_tau(next_progress):.3f}")

    # Save final model
    out = "/tmp/rlvr_gym/final_model_v2b"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\n[OK] Model saved to {out}")

    # Final eval
    print("\n[*] Final eval on 10 mixed tasks...")
    from task_generator import Task
    eval_tasks = [generate_task(d) for d in [1,1,1,2,2,2,3,3,3,3]]
    correct = 0
    for i, etask in enumerate(eval_tasks):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": etask.instruction},
        ]
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
        outputs = trainer.model.generate(
            inputs.to(trainer.model.device),
            max_new_tokens=384, temperature=0.1, do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        result = execute_and_verify(etask, response)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{status}] Task {i+1} (d={etask.difficulty} {etask.task_type}): {result.steps_completed}/{result.steps_total}")
        if result.success:
            correct += 1

    print(f"\n{'='*60}")
    print(f"  FINAL v2b: {correct}/10 = {correct*10}% accuracy")
    print(f"{'='*60}")

    # Print curriculum evolution
    print(f"\n[*] Curriculum evolution:")
    for ep, rewards in enumerate(all_rewards_by_epoch):
        for diff in [1, 2, 3]:
            avg = sum(rewards[diff]) / len(rewards[diff]) if rewards[diff] else 0
            print(f"  Epoch {ep+1}, Difficulty {diff}: {avg:.3f}")

if __name__ == "__main__":
    api = start_api()
    try:
        run_training()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        api.terminate()
        api.wait()
        print("[*] API server stopped")
