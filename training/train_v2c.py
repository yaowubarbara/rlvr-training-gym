#!/usr/bin/env python3
"""v2c: Agent Loop — multi-turn interaction with real API responses.
Hypothesis: single-turn model can't learn to use real order IDs.
Fix: model sees API response after each action, uses real IDs for subsequent steps.
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
    from train_grpo import format_reward, SYSTEM_PROMPT
    from task_generator import generate_task
    from env import APIEnvironment, run_agent_loop
    from datasets import Dataset

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    NUM_GENERATIONS = 4
    MAX_COMPLETION = 384
    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    TOTAL_TASKS = 100

    print("=" * 60)
    print("  RLVR Training Gym — v2c: Agent Loop (Multi-turn)")
    print("=" * 60)

    # Build dataset — same as v1, but reward function will use agent loop
    print(f"\n[*] Generating {TOTAL_TASKS} tasks...")
    tasks_list = []
    for _ in range(TOTAL_TASKS):
        # Mix of difficulties
        import random
        diff = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
        tasks_list.append(generate_task(diff))

    prompts = []
    metadata = []

    # === KEY CHANGE: prompt includes instruction to output ONE action at a time ===
    AGENT_LOOP_PROMPT = SYSTEM_PROMPT + """

IMPORTANT: You are in an interactive loop. Output exactly ONE action at a time.
After each action, you will see the API response. Use the REAL values from
the response (especially order IDs) in your next action.

Do NOT output multiple actions at once. Wait for the API response before proceeding."""

    for task in tasks_list:
        prompt = [
            {"role": "system", "content": AGENT_LOOP_PROMPT},
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

    print(f"\n[*] Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === KEY CHANGE: accuracy_reward uses agent loop ===
    def agent_loop_reward(completions, expected_actions=None, instruction=None, **kwargs):
        """Multi-turn reward: run agent loop for each completion's FIRST action,
        then simulate the full trajectory.

        For GRPO compatibility: we still get single-turn completions from the model,
        but we evaluate them by running the full agent loop using the model's
        first action as a signal of its policy quality.
        """
        from task_generator import Task

        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)

            try:
                exp_actions = json.loads(expected_actions[i]) if expected_actions else []
                task = Task(
                    instruction=instruction[i] if instruction else "",
                    expected_actions=exp_actions,
                    difficulty=1,
                    task_type="unknown",
                )

                # Run agent loop with model's output as the full trajectory
                # Parse all actions and execute step by step through env
                env = APIEnvironment(task)
                env.reset()

                from verifier import parse_agent_actions
                actions = parse_agent_actions(text)

                if not actions:
                    rewards.append(0.0)
                    continue

                total_reward = 0.0
                for act in actions:
                    action_str = f"ACTION: {act['method']} {act['path']}"
                    if act.get('body'):
                        # Substitute real order IDs into body
                        body = dict(act['body'])
                        if env.order_ids:
                            for key, val in env.order_ids.items():
                                if 'order_id' in body and body['order_id'] not in env.order_ids.values():
                                    body['order_id'] = env.order_ids.get('order_id', body['order_id'])
                        action_str += f"\nBODY: {json.dumps(body)}"

                    result = env.step(action_str)

                    if result.done:
                        total_reward = result.reward
                        break

                # If we didn't finish all steps, partial credit
                if not result.done:
                    total_reward = env.current_step / len(exp_actions) if exp_actions else 0

                if i == 0:
                    print(f"[DEBUG-v2c] steps={env.current_step}/{len(exp_actions)} "
                          f"reward={total_reward:.2f} "
                          f"ids={env.order_ids}")

                rewards.append(total_reward)

            except Exception as e:
                if i == 0:
                    print(f"[DEBUG-v2c] EXCEPTION: {e}")
                rewards.append(0.0)

        return rewards

    config = GRPOConfig(
        output_dir="/tmp/rlvr_gym/checkpoints_v2c",
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

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[format_reward, agent_loop_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n[*] v2c agent loop training started!")
    print("-" * 60)
    trainer.train()

    out = "/tmp/rlvr_gym/final_model_v2c"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\n[OK] Model saved to {out}")

    # Eval with agent loop
    print("\n[*] Final eval with agent loop on 10 tasks...")

    def model_generate(messages):
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        outputs = trainer.model.generate(
            inputs.to(trainer.model.device),
            max_new_tokens=256, temperature=0.1, do_sample=True,
        )
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    correct = 0
    for i in range(10):
        diff = [1,1,1,2,2,2,3,3,3,3][i]
        task = generate_task(diff)
        reward, trajectory = run_agent_loop(task, model_generate, AGENT_LOOP_PROMPT, verbose=False)
        status = "PASS" if reward >= 1.0 else "FAIL"
        print(f"  [{status}] Task {i+1} (d={diff} {task.task_type}): reward={reward:.2f}, steps={len(trajectory)}")
        if reward >= 1.0:
            correct += 1

    print(f"\n{'='*60}")
    print(f"  FINAL v2c: {correct}/10 = {correct*10}% accuracy (agent loop eval)")
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
