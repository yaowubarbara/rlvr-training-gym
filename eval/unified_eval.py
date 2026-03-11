#!/usr/bin/env python3
"""Unified eval script — outputs all metrics for binary vs partial reward comparison.

Metrics:
  1. overall_success (pass rate)
  2. D1/D2/D3 success rates
  3. avg_partial_progress (steps_completed / steps_total)
  4. avg_steps_completed per difficulty
  5. final_state_verification_failure_rate (steps all matched but final state wrong)
  6. early_stop_rate ("Expected N actions, got M" where M < N)
  7. Per-task failure taxonomy: early_stop / param_error / multi_entity_fail

Usage:
  python3 /tmp/unified_eval.py --model /path/to/model --seed 42 --tasks-per-diff 10
  python3 /tmp/unified_eval.py --model /path/to/model --seed 42 --tasks-per-diff 10 --output /tmp/eval_result.json
"""
import argparse, json, os, sys, time, random
import numpy as np

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_DIR = "/tmp/rlvr_gym"
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)


def safe_generate(model, tokenizer, messages, max_new_tokens=384, temperature=0.1):
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_len = input_ids.shape[1]
    outputs = model.generate(
        input_ids.to(model.device),
        max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True,
    )
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def classify_failure(task, result, num_actions_generated):
    """Classify failure into taxonomy aligned with reward hacking literature.

    Two primary categories for binary vs partial reward comparison:
      Type A — "high_partial_low_final": partial progress >= 0.75 but success=False
        → suboptimal policy convergence (does most steps, fails final verification)
        → maps to: "On Designing Effective RL Reward" process reward harm
      Type B — "early_stop": generated fewer actions than needed
        → partial-credit induced early stopping (gets enough reward, stops)
        → maps to: GRPO reward hacking / MO-GRPO variance-seeking

    Secondary (for completeness):
      multi_entity_fail: multi-order tasks where entity tracking fails
      param_error: enough actions but wrong params (not early stop, not high partial)
    """
    steps_total = result.steps_total
    steps_completed = result.steps_completed
    partial = steps_completed / steps_total if steps_total > 0 else 0.0

    if result.success:
        return "success"

    # Type B: Early stop — model generated fewer actions than expected
    if num_actions_generated < steps_total:
        return "early_stop"

    # Type A: High partial, low final — did most/all steps but verification failed
    if partial >= 0.75 and not result.success:
        return "high_partial_low_final"

    # Multi-entity management failure
    if "multi_order" in task.task_type and steps_completed < steps_total:
        return "multi_entity_fail"

    # Generic param error (low partial, not early stop)
    if steps_completed < steps_total:
        return "param_error"

    return "other"


def run_eval(model_path, seed, tasks_per_diff, output_path=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from task_generator import generate_task
    from train_grpo import SYSTEM_PROMPT
    from verifier import execute_and_verify, parse_agent_actions

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()

    # Generate fixed eval tasks
    difficulties = []
    for d in [1, 2, 3]:
        difficulties.extend([d] * tasks_per_diff)

    tasks = [generate_task(d) for d in difficulties]

    results_list = []

    print(f"[*] Evaluating {len(tasks)} tasks (seed={seed})...")
    with torch.no_grad():
        for i, task in enumerate(tasks):
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ]
            t0 = time.time()
            response = safe_generate(model, tokenizer, prompt)
            gen_time = time.time() - t0

            result = execute_and_verify(task, response)
            actions = parse_agent_actions(response)
            num_actions = len(actions)

            failure_mode = classify_failure(task, result, num_actions)

            entry = {
                "task_idx": i,
                "difficulty": task.difficulty,
                "task_type": task.task_type,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "steps_total": result.steps_total,
                "partial_reward": result.partial_reward,
                "num_actions_generated": num_actions,
                "failure_mode": failure_mode,
                "errors": result.errors[:3],
                "gen_time": round(gen_time, 2),
                "response_preview": response[:200],
            }
            results_list.append(entry)

            status = "PASS" if result.success else "FAIL"
            print(f"  [{status}] Task {i+1} (d={task.difficulty} {task.task_type}): "
                  f"{result.steps_completed}/{result.steps_total} "
                  f"actions={num_actions} mode={failure_mode}")

    # === Aggregate metrics ===
    metrics = compute_metrics(results_list)
    metrics["model_path"] = model_path
    metrics["seed"] = seed
    metrics["tasks_per_diff"] = tasks_per_diff
    metrics["total_tasks"] = len(tasks)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  UNIFIED EVAL — {model_path.split('/')[-1]}")
    print(f"  Seed={seed}, Tasks/diff={tasks_per_diff}")
    print(f"{'='*60}")
    print(f"  Overall success: {metrics['overall_success']:.1%}")
    print(f"  D1 success: {metrics['d1_success']:.1%}")
    print(f"  D2 success: {metrics['d2_success']:.1%}")
    print(f"  D3 success: {metrics['d3_success']:.1%}")
    print(f"  Avg partial progress: {metrics['avg_partial_progress']:.3f}")
    print(f"  Avg steps completed (D3): {metrics['d3_avg_steps_completed']:.2f}/{metrics['d3_avg_steps_total']:.1f}")
    print(f"  Final-state verification failure rate: {metrics['final_state_fail_rate']:.1%}")
    print(f"  Early stop rate (all): {metrics['early_stop_rate']:.1%}")
    print(f"  Early stop rate (D3): {metrics['d3_early_stop_rate']:.1%}")
    # Literature-aligned reward hacking indicators
    print(f"\n  === Reward Hacking Indicators (literature-aligned) ===")
    print(f"  Type A 'high partial, low final' rate: {metrics['type_a_rate']:.1%} "
          f"(suboptimal policy convergence)")
    print(f"  Type B 'early stop' rate: {metrics['type_b_rate']:.1%} "
          f"(partial-credit induced stopping)")
    print(f"  D3 Type A: {metrics['d3_type_a_rate']:.1%}")
    print(f"  D3 Type B: {metrics['d3_type_b_rate']:.1%}")

    print(f"\n  Full failure taxonomy (all failures):")
    for mode, count in sorted(metrics['failure_taxonomy'].items(), key=lambda x: -x[1]):
        pct = count / max(metrics['total_failures'], 1)
        print(f"    {mode}: {count} ({pct:.0%})")
    print(f"\n  D3 failure taxonomy:")
    for mode, count in sorted(metrics['d3_failure_taxonomy'].items(), key=lambda x: -x[1]):
        pct = count / max(metrics['d3_total_failures'], 1)
        print(f"    {mode}: {count} ({pct:.0%})")

    # Save
    output = {
        "metrics": metrics,
        "per_task": results_list,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {output_path}")

    # Always save to a default location too
    default_path = f"/tmp/rlvr_gym/eval_{model_path.split('/')[-1]}_seed{seed}.json"
    with open(default_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[OK] Also saved to {default_path}")

    return output


def compute_metrics(results_list):
    total = len(results_list)
    successes = [r for r in results_list if r["success"]]
    failures = [r for r in results_list if not r["success"]]

    by_diff = {1: [], 2: [], 3: []}
    for r in results_list:
        by_diff[r["difficulty"]].append(r)

    def success_rate(items):
        if not items:
            return 0.0
        return sum(1 for r in items if r["success"]) / len(items)

    def avg_partial(items):
        if not items:
            return 0.0
        return sum(r["partial_reward"] for r in items) / len(items)

    def avg_steps(items):
        if not items:
            return 0.0
        return sum(r["steps_completed"] for r in items) / len(items)

    def avg_total_steps(items):
        if not items:
            return 0.0
        return sum(r["steps_total"] for r in items) / len(items)

    # Final-state verification failures: steps_completed == steps_total but not success
    final_state_fails = [r for r in results_list
                         if r["steps_completed"] == r["steps_total"] and not r["success"]]

    # Early stops: generated fewer actions than expected
    early_stops = [r for r in results_list
                   if r["num_actions_generated"] < r["steps_total"] and not r["success"]]

    d3_early_stops = [r for r in by_diff[3]
                      if r["num_actions_generated"] < r["steps_total"] and not r["success"]]

    # Failure taxonomy
    taxonomy = {}
    d3_taxonomy = {}
    for r in failures:
        mode = r["failure_mode"]
        taxonomy[mode] = taxonomy.get(mode, 0) + 1
        if r["difficulty"] == 3:
            d3_taxonomy[mode] = d3_taxonomy.get(mode, 0) + 1

    d3_failures = [r for r in by_diff[3] if not r["success"]]

    # Literature-aligned: Type A (high partial, low final) and Type B (early stop)
    type_a = [r for r in failures if r["failure_mode"] == "high_partial_low_final"]
    type_b = [r for r in failures if r["failure_mode"] == "early_stop"]
    d3_type_a = [r for r in d3_failures if r["failure_mode"] == "high_partial_low_final"]
    d3_type_b = [r for r in d3_failures if r["failure_mode"] == "early_stop"]

    return {
        "overall_success": success_rate(results_list),
        "d1_success": success_rate(by_diff[1]),
        "d2_success": success_rate(by_diff[2]),
        "d3_success": success_rate(by_diff[3]),
        "avg_partial_progress": avg_partial(results_list),
        "d1_avg_partial": avg_partial(by_diff[1]),
        "d2_avg_partial": avg_partial(by_diff[2]),
        "d3_avg_partial": avg_partial(by_diff[3]),
        "d3_avg_steps_completed": avg_steps(by_diff[3]),
        "d3_avg_steps_total": avg_total_steps(by_diff[3]),
        "final_state_fail_rate": len(final_state_fails) / max(total, 1),
        "early_stop_rate": len(early_stops) / max(total, 1),
        "d3_early_stop_rate": len(d3_early_stops) / max(len(by_diff[3]), 1),
        "total_failures": len(failures),
        "d3_total_failures": len(d3_failures),
        "failure_taxonomy": taxonomy,
        "d3_failure_taxonomy": d3_taxonomy,
        # Literature-aligned reward hacking rates
        "type_a_rate": len(type_a) / max(total, 1),
        "type_b_rate": len(type_b) / max(total, 1),
        "d3_type_a_rate": len(d3_type_a) / max(len(by_diff[3]), 1),
        "d3_type_b_rate": len(d3_type_b) / max(len(by_diff[3]), 1),
        "type_a_count": len(type_a),
        "type_b_count": len(type_b),
        "d3_type_a_count": len(d3_type_a),
        "d3_type_b_count": len(d3_type_b),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks-per-diff", type=int, default=10, help="Tasks per difficulty level")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    run_eval(args.model, args.seed, args.tasks_per_diff, args.output)
