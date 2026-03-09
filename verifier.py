"""
API Training Gym - Module 3: Deterministic Verifier
Provides binary rewards (0/1) by checking if agent's API calls are correct.
This is the core of RLVR - verifiable rewards without learned reward models.
"""
import json
import re
import time
import httpx
from dataclasses import dataclass
from task_generator import Task

API_BASE = "http://127.0.0.1:8000"


def _http_request(method, url, json_body=None, retries=5):
    """HTTP request with retry on 502/503/connection errors."""
    for attempt in range(retries):
        try:
            if method == "GET":
                resp = httpx.get(url, timeout=10)
            elif method == "POST":
                if json_body is not None:
                    resp = httpx.post(url, json=json_body, timeout=10)
                else:
                    resp = httpx.post(url, timeout=10)
            else:
                return None
            if resp.status_code in (502, 503) and attempt < retries - 1:
                time.sleep(1.0)
                continue
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < retries - 1:
                time.sleep(1.0)
                continue
            raise
    return resp


@dataclass
class VerificationResult:
    """Result of verifying an agent's action sequence."""
    reward: float              # 0.0 or 1.0 (binary RLVR reward)
    partial_reward: float      # 0.0 to 1.0 (fraction of correct steps, for analysis)
    success: bool              # Did the full task complete correctly?
    steps_completed: int       # How many steps were correct
    steps_total: int           # Total expected steps
    errors: list[str]          # What went wrong
    api_responses: list[dict]  # Raw API responses for debugging


def parse_agent_actions(agent_output: str) -> list[dict]:
    """Parse the LLM agent's output into structured API calls.

    Expected format from agent:
    ```
    ACTION: POST /orders
    BODY: {"customer_name": "Alice", "product": "Laptop", "quantity": 1, "price": 999.99}

    ACTION: POST /orders/abc123/pay
    BODY: {"order_id": "abc123", "amount": 999.99, "method": "card"}
    ```
    """
    actions = []
    current_action = {}

    for line in agent_output.strip().split("\n"):
        line = line.strip()

        if line.startswith("ACTION:"):
            if current_action:
                actions.append(current_action)
            parts = line[7:].strip().split(" ", 1)
            current_action = {
                "method": parts[0].upper(),
                "path": parts[1] if len(parts) > 1 else "",
                "body": None,
            }

        elif line.startswith("BODY:"):
            body_str = line[5:].strip()
            try:
                current_action["body"] = json.loads(body_str)
            except json.JSONDecodeError:
                current_action["body"] = {"_parse_error": body_str}

    if current_action:
        actions.append(current_action)

    return actions


def execute_and_verify(task: Task, agent_output: str) -> VerificationResult:
    """Execute agent's actions against the API and verify correctness.

    This is the RLVR reward function:
    - reward=1.0 if ALL actions are correct and task completes
    - reward=0.0 if ANY action fails or sequence is wrong
    - partial_reward gives credit for partially correct sequences (for analysis only)
    """
    errors = []
    api_responses = []
    steps_completed = 0
    order_ids = {}  # Track created order IDs for substitution

    # Parse agent output
    agent_actions = parse_agent_actions(agent_output)

    if not agent_actions:
        return VerificationResult(
            reward=0.0, partial_reward=0.0, success=False,
            steps_completed=0, steps_total=len(task.expected_actions),
            errors=["No actions parsed from agent output"],
            api_responses=[],
        )

    # Reset the API system
    try:
        _http_request("POST", f"{API_BASE}/reset")
    except Exception as e:
        return VerificationResult(
            reward=0.0, partial_reward=0.0, success=False,
            steps_completed=0, steps_total=len(task.expected_actions),
            errors=[f"Could not reset API: {e}"],
            api_responses=[],
        )

    expected = task.expected_actions

    # Check step count
    if len(agent_actions) != len(expected):
        errors.append(f"Expected {len(expected)} actions, got {len(agent_actions)}")

    # Execute each action
    n_check = min(len(agent_actions), len(expected))
    for i in range(n_check):
        exp = expected[i]
        act = agent_actions[i]

        # Check method
        if act["method"] != exp["method"]:
            errors.append(f"Step {i+1}: Expected {exp['method']}, got {act['method']}")
            break

        # Check path pattern (allow dynamic order_id)
        exp_path_raw = exp["path"]  # Keep original template for wildcard matching
        act_path = act["path"]

        # Use ORIGINAL template for structural matching (before substitution)
        exp_parts_raw = exp_path_raw.split("/")
        act_parts = act_path.split("/")

        path_ok = len(exp_parts_raw) == len(act_parts)
        if path_ok:
            for ep, ap in zip(exp_parts_raw, act_parts):
                if ep.startswith("{") and ep.endswith("}"):
                    continue  # Dynamic part — accept any value from agent
                if ep != ap:
                    path_ok = False
                    break

        if not path_ok:
            errors.append(f"Step {i+1}: Expected path ~{exp_path_raw}, got {act_path}")
            break

        # Build execution path: replace agent's placeholder IDs with real ones
        exec_path = act_path
        if order_ids:
            exec_parts = exec_path.split("/")
            for j, (ep, xp) in enumerate(zip(exp_parts_raw, exec_parts)):
                if ep.startswith("{") and ep.endswith("}"):
                    key = ep[1:-1]
                    if key in order_ids:
                        exec_parts[j] = order_ids[key]
            exec_path = "/".join(exec_parts)

        # Also substitute order IDs in request body
        exec_body = act["body"]
        if exec_body and order_ids:
            body_str = json.dumps(exec_body)
            for key, oid in order_ids.items():
                # Replace both template vars and agent's placeholder values
                body_str = body_str.replace(f"{{{key}}}", oid)
            # Replace any non-real order_id values in body's "order_id" field
            if "order_id" in (exec_body or {}):
                agent_oid = exec_body.get("order_id", "")
                if agent_oid and agent_oid not in order_ids.values():
                    body_str = body_str.replace(f'"{agent_oid}"', f'"{order_ids.get("order_id", agent_oid)}"')
            try:
                exec_body = json.loads(body_str)
            except json.JSONDecodeError:
                pass

        # Execute the API call (using real IDs)
        try:
            url = f"{API_BASE}{exec_path}"
            resp = _http_request(act["method"], url, exec_body)
            if resp is None:
                errors.append(f"Step {i+1}: Unknown method {act['method']}")
                break

            api_responses.append({
                "status": resp.status_code,
                "body": resp.json() if resp.status_code < 400 else resp.text,
            })

            if resp.status_code >= 400:
                errors.append(f"Step {i+1}: API returned {resp.status_code}: {resp.text[:100]}")
                break

            # Track order IDs from create responses
            if act["path"] == "/orders" and act["method"] == "POST" and resp.status_code == 200:
                resp_data = resp.json()
                order_num = len(order_ids) + 1
                order_ids["order_id"] = resp_data["id"]
                order_ids[f"order_id_{order_num}"] = resp_data["id"]

            steps_completed += 1

        except Exception as e:
            errors.append(f"Step {i+1}: Execution error: {e}")
            break

    # Calculate rewards
    steps_total = len(expected)
    partial = steps_completed / steps_total if steps_total > 0 else 0.0
    success = steps_completed == steps_total and len(errors) == 0

    # Verify final state if task completed
    if success and order_ids:
        try:
            main_id = order_ids.get("order_id", "")
            if main_id:
                resp = httpx.get(f"{API_BASE}/orders/{main_id}", timeout=5)
                if resp.status_code == 200:
                    final_state = resp.json()
                    # Check final state matches expected
                    expected_final = _get_expected_final_state(task)
                    if expected_final and final_state.get("status") != expected_final:
                        errors.append(
                            f"Final state: expected '{expected_final}', "
                            f"got '{final_state.get('status')}'"
                        )
                        success = False
        except Exception:
            pass

    return VerificationResult(
        reward=1.0 if success else 0.0,
        partial_reward=round(partial, 2),
        success=success,
        steps_completed=steps_completed,
        steps_total=steps_total,
        errors=errors,
        api_responses=api_responses,
    )


def _get_expected_final_state(task: Task) -> str | None:
    """Determine expected final order state from task type."""
    state_map = {
        "create_order": "created",
        "create_and_check": "created",
        "create_and_cancel": "cancelled",
        "create_and_pay": "paid",
        "create_pay_ship": "shipped",
        "pay_and_refund": "refunded",
        "full_delivery": "delivered",
        "full_lifecycle_with_refund": "refunded",
    }
    return state_map.get(task.task_type)


def batch_verify(tasks: list[Task], agent_outputs: list[str]) -> dict:
    """Verify a batch of tasks and return aggregate metrics."""
    results = []
    for task, output in zip(tasks, agent_outputs):
        result = execute_and_verify(task, output)
        results.append(result)

    total = len(results)
    successes = sum(1 for r in results if r.success)
    avg_partial = sum(r.partial_reward for r in results) / total if total > 0 else 0

    # Per-difficulty breakdown
    by_difficulty = {}
    for task, result in zip(tasks, results):
        d = task.difficulty
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "success": 0}
        by_difficulty[d]["total"] += 1
        if result.success:
            by_difficulty[d]["success"] += 1

    return {
        "total_tasks": total,
        "successes": successes,
        "success_rate": round(successes / total, 3) if total > 0 else 0,
        "avg_partial_reward": round(avg_partial, 3),
        "by_difficulty": {
            d: {**v, "rate": round(v["success"] / v["total"], 3)}
            for d, v in sorted(by_difficulty.items())
        },
        "results": results,
    }


if __name__ == "__main__":
    # Demo: test with a mock agent output
    from task_generator import generate_task

    task = generate_task(1)
    print(f"Task: {task.instruction}")
    print(f"Expected actions: {len(task.expected_actions)}")

    # Simulate correct agent output
    mock_output = "ACTION: POST /orders\n"
    if task.expected_actions[0].get("body"):
        mock_output += f"BODY: {json.dumps(task.expected_actions[0]['body'])}\n"
    print(f"\nMock agent output:\n{mock_output}")

    # Note: requires API server running on port 8000
    print("\n[!] Run 'uvicorn api_system:app --port 8000' first to test verification")
