#!/usr/bin/env python3
"""v3a: SFT Baseline — supervised fine-tuning on correct demonstrations.
Hypothesis: How much does RL (GRPO) add over pure imitation learning?
This gives us the SFT baseline to compare against v1 GRPO.
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

    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from train_grpo import SYSTEM_PROMPT
    from task_generator import generate_task
    from datasets import Dataset
    import torch
    import random

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    TOTAL_TASKS = 300  # Same total training examples as v1 (100 tasks × 3 epochs)

    print("=" * 60)
    print("  RLVR Training Gym — v3a: SFT Baseline")
    print("=" * 60)

    # Generate training data: (instruction, correct_output) pairs
    print(f"\n[*] Generating {TOTAL_TASKS} SFT training examples...")
    random.seed(42)

    sft_data = []
    for i in range(TOTAL_TASKS):
        # Same difficulty distribution as v1
        diff = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
        task = generate_task(diff)

        # Build the "correct" output from expected_actions
        correct_output = ""
        for act in task.expected_actions:
            correct_output += f"ACTION: {act['method']} {act['path']}\n"
            if act.get("body"):
                correct_output += f"BODY: {json.dumps(act['body'])}\n"

        # Build full conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
            {"role": "assistant", "content": correct_output.strip()},
        ]

        sft_data.append({
            "messages": messages,
            "difficulty": diff,
            "task_type": task.task_type,
        })

    dist = {1: 0, 2: 0, 3: 0}
    for d in sft_data:
        dist[d["difficulty"]] += 1
    print(f"[OK] SFT data: {len(sft_data)} examples (easy={dist[1]}, med={dist[2]}, hard={dist[3]})")

    print(f"\n[*] Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    # Tokenize for causal LM training
    print("[*] Tokenizing...")

    def tokenize_messages(example):
        """Tokenize a conversation for causal LM training.
        Mask the system+user tokens so we only train on assistant output.
        """
        messages = example["messages"]
        # Full conversation
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = tokenizer(full, truncation=True, max_length=512)["input_ids"]

        # Everything except assistant response (for masking)
        prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_only, truncation=True, max_length=512)["input_ids"]

        # Labels: -100 for prompt tokens (masked), actual ids for assistant tokens
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        # Pad labels to match input length
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    # Create dataset
    dataset = Dataset.from_list(sft_data)
    tokenized = dataset.map(tokenize_messages, remove_columns=dataset.column_names)

    # Data collator that pads
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir="/tmp/rlvr_gym/checkpoints_v3a",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # Higher LR for SFT (vs 1e-6 for GRPO)
        logging_steps=10,
        save_steps=100,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(f"\n[*] SFT training started!")
    print(f"    {len(tokenized)} examples × 3 epochs")
    print("-" * 60)
    trainer.train()

    out = "/tmp/rlvr_gym/final_model_v3a"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\n[OK] Model saved to {out}")

    # Eval using persistent eval set
    print("\n[*] Running persistent evaluation...")
    os.system(f"{PYTHON} /tmp/eval_persistent.py {out} v3a_sft")

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
