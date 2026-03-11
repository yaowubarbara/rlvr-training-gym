#!/usr/bin/env python3
"""v2a: Hard-only training — remove easy tasks that saturate reward.
Hypothesis: v1 reward saturates because easy tasks are too simple.
Fix: only train on difficulty 2-3 tasks to keep gradient signal alive.
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
    from task_generator import generate_task
    from datasets import Dataset

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    NUM_GENERATIONS = 4
    MAX_COMPLETION = 384
    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    TOTAL_TASKS = 100

    print("=" * 60)
    print("  RLVR Training Gym — v2a: Hard-Only (difficulty 2-3)")
    print("=" * 60)

    # === KEY CHANGE: only difficulty 2-3 tasks ===
    print(f"\n[*] Generating {TOTAL_TASKS} HARD tasks (difficulty 2-3 only)...")
    tasks = []
    while len(tasks) < TOTAL_TASKS:
        diff = 2 if len(tasks) % 2 == 0 else 3  # 50/50 split
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

    med = sum(1 for d in dataset['difficulty'] if d == 2)
    hard = sum(1 for d in dataset['difficulty'] if d == 3)
    print(f"[OK] Dataset: {len(dataset)} tasks (easy=0, med={med}, hard={hard})")

    print(f"\n[*] Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir="/tmp/rlvr_gym/checkpoints_v2a",
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-6,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION,
        beta=0.04,
        logging_steps=1,
        save_steps=50,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        temperature=0.9,
        max_grad_norm=1.0,
        per_device_eval_batch_size=1,
    )

    steps_per_epoch = TOTAL_TASKS // (BATCH_SIZE * GRAD_ACCUM)
    print(f"[*] Config: gen={NUM_GENERATIONS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"    Steps/epoch: ~{steps_per_epoch}, total: ~{steps_per_epoch * 3}")

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[format_reward, accuracy_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n[*] v2a training started!")
    print("-" * 60)
    trainer.train()

    out = "/tmp/rlvr_gym/final_model_v2a"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\n[OK] Model saved to {out}")

    # Quick eval
    print("\n[*] Quick eval on 10 test tasks (mixed difficulty)...")
    from verifier import execute_and_verify
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
    print(f"  FINAL v2a: {correct}/10 = {correct*10}% accuracy")
    print(f"{'='*60}")

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
