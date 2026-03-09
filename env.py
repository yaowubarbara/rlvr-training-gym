"""
API Training Gym - Module 5: Agent Loop Environment
Converts single-turn verifier into multi-turn step-by-step environment.

Before (single-turn):
  model → "ACTION: POST /orders\nBODY: {...}\nACTION: POST /orders/abc123/pay\n..."
  verifier → reward 0 or 1

After (agent loop):
  model → "ACTION: POST /orders\nBODY: {...}"
  env   → {"status": 200, "body": {"id": "ord_7f3a", "status": "created", ...}}
  model → "ACTION: POST /orders/ord_7f3a/pay\nBODY: {...}"   ← uses REAL id!
  env   → {"status": 200, "body": {"status": "paid", ...}}
  env   → reward=1.0, done=True
"""
import json
import httpx
from dataclasses import dataclass, field
from task_generator import Task
from verifier import parse_agent_actions, _http_request

API_BASE = "http://127.0.0.1:8000"


@dataclass
class StepResult:
    observation: str       # What the model sees (API response as text)
    reward: float          # 0.0 during steps, final reward at done
    done: bool             # Episode finished?
    success: bool          # Task completed correctly?
    info: dict = field(default_factory=dict)


class APIEnvironment:
    """Step-by-step environment for agent loop training.

    Usage:
        env = APIEnvironment(task)
        obs = env.reset()
        while True:
            action = model.generate(obs)  # model outputs 1 ACTION
            result = env.step(action)
            obs = result.observation
            if result.done:
                final_reward = result.reward
                break
    """

    def __init__(self, task: Task, max_steps: int = 10):
        self.task = task
        self.max_steps = max_steps
        self.expected = task.expected_actions
        self.current_step = 0
        self.history: list[dict] = []   # [(action, observation), ...]
        self.order_ids: dict = {}
        self.errors: list[str] = []

    def reset(self) -> str:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.history = []
        self.order_ids = {}
        self.errors = []

        # Reset API state
        try:
            _http_request("POST", f"{API_BASE}/reset")
        except Exception as e:
            return f"[ERROR] Could not reset API: {e}"

        # Initial observation = task instruction
        return self.task.instruction

    def step(self, agent_output: str) -> StepResult:
        """Execute one agent action and return observation.

        Args:
            agent_output: Model's text output containing exactly ONE action.
                         e.g. "ACTION: POST /orders\nBODY: {...}"
        """
        self.current_step += 1

        # Parse the single action
        actions = parse_agent_actions(agent_output)
        if not actions:
            return StepResult(
                observation="[ERROR] No valid ACTION found in your output. Use format: ACTION: METHOD /path\nBODY: {...}",
                reward=0.0, done=True, success=False,
                info={"error": "parse_failed"}
            )

        # Take only the first action (ignore extras)
        act = actions[0]

        # Check if we've exceeded expected steps
        if self.current_step > len(self.expected):
            return StepResult(
                observation=f"[ERROR] Task only requires {len(self.expected)} steps. You've done {self.current_step}.",
                reward=0.0, done=True, success=False,
                info={"error": "too_many_steps"}
            )

        # Validate against expected action
        exp = self.expected[self.current_step - 1]

        # Check method
        if act["method"] != exp["method"]:
            self.errors.append(f"Step {self.current_step}: Expected {exp['method']}, got {act['method']}")
            return StepResult(
                observation=f"[ERROR] Wrong method. Expected {exp['method']}.",
                reward=0.0, done=True, success=False,
                info={"error": "wrong_method"}
            )

        # Check path structure (using template matching)
        exp_parts = exp["path"].split("/")
        act_parts = act["path"].split("/")
        path_ok = len(exp_parts) == len(act_parts)
        if path_ok:
            for ep, ap in zip(exp_parts, act_parts):
                if ep.startswith("{") and ep.endswith("}"):
                    continue
                if ep != ap:
                    path_ok = False
                    break

        if not path_ok:
            self.errors.append(f"Step {self.current_step}: Wrong path structure")
            return StepResult(
                observation=f"[ERROR] Wrong endpoint path.",
                reward=0.0, done=True, success=False,
                info={"error": "wrong_path"}
            )

        # Substitute real order IDs into path
        exec_path = act["path"]
        if self.order_ids:
            parts = exec_path.split("/")
            for j, (ep, xp) in enumerate(zip(exp_parts, parts)):
                if ep.startswith("{") and ep.endswith("}"):
                    key = ep[1:-1]
                    if key in self.order_ids:
                        parts[j] = self.order_ids[key]
            exec_path = "/".join(parts)

        # Substitute real order IDs into body
        exec_body = act["body"]
        if exec_body and self.order_ids:
            body_str = json.dumps(exec_body)
            for key, oid in self.order_ids.items():
                body_str = body_str.replace(f"{{{key}}}", oid)
            if "order_id" in (exec_body or {}):
                agent_oid = exec_body.get("order_id", "")
                if agent_oid and agent_oid not in self.order_ids.values():
                    real_id = self.order_ids.get("order_id", agent_oid)
                    body_str = body_str.replace(f'"{agent_oid}"', f'"{real_id}"')
            try:
                exec_body = json.loads(body_str)
            except json.JSONDecodeError:
                pass

        # Execute the API call
        try:
            url = f"{API_BASE}{exec_path}"
            resp = _http_request(act["method"], url, exec_body)

            if resp is None:
                return StepResult(
                    observation=f"[ERROR] Unknown HTTP method: {act['method']}",
                    reward=0.0, done=True, success=False,
                    info={"error": "unknown_method"}
                )

            # Build observation (what the model sees)
            if resp.status_code < 400:
                resp_data = resp.json()
                observation = json.dumps(resp_data, indent=2)

                # Track order IDs
                if act["path"] == "/orders" and act["method"] == "POST" and resp.status_code == 200:
                    order_num = len(self.order_ids) + 1
                    self.order_ids["order_id"] = resp_data["id"]
                    self.order_ids[f"order_id_{order_num}"] = resp_data["id"]
            else:
                observation = f"[API ERROR {resp.status_code}] {resp.text[:200]}"
                self.errors.append(f"Step {self.current_step}: API returned {resp.status_code}")
                return StepResult(
                    observation=observation,
                    reward=0.0, done=True, success=False,
                    info={"error": "api_error", "status": resp.status_code}
                )

        except Exception as e:
            return StepResult(
                observation=f"[ERROR] Request failed: {e}",
                reward=0.0, done=True, success=False,
                info={"error": "request_failed"}
            )

        # Record history
        self.history.append({
            "action": f"{act['method']} {act['path']}",
            "body": act["body"],
            "observation": observation,
        })

        # Check if task is complete
        done = self.current_step >= len(self.expected)
        success = done and len(self.errors) == 0

        # Reward: partial during episode, full at end
        if done:
            reward = 1.0 if success else 0.0
        else:
            reward = 0.0  # no intermediate reward (sparse)

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            success=success,
            info={
                "step": self.current_step,
                "total_steps": len(self.expected),
                "order_ids": dict(self.order_ids),
            }
        )

    def get_conversation_history(self) -> list[dict]:
        """Build chat messages from history (for model context)."""
        messages = [
            {"role": "user", "content": self.task.instruction},
        ]
        for h in self.history:
            # Agent's action
            action_text = f"ACTION: {h['action']}"
            if h["body"]:
                action_text += f"\nBODY: {json.dumps(h['body'])}"
            messages.append({"role": "assistant", "content": action_text})
            # Environment observation
            messages.append({"role": "user", "content": f"API Response:\n{h['observation']}"})
        return messages


def run_agent_loop(task: Task, generate_fn, system_prompt: str, max_steps: int = 10, verbose: bool = False):
    """Run a complete agent loop episode.

    Args:
        task: The task to complete
        generate_fn: Function(messages) -> str that generates model output
        system_prompt: System prompt for the model
        max_steps: Maximum steps before termination
        verbose: Print each step

    Returns:
        (reward, trajectory) tuple
    """
    env = APIEnvironment(task, max_steps=max_steps)
    obs = env.reset()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs},
    ]

    trajectory = []

    for step in range(max_steps):
        # Generate one action
        response = generate_fn(messages)

        if verbose:
            print(f"  Step {step+1}: {response[:100]}...")

        # Execute in environment
        result = env.step(response)

        if verbose:
            print(f"  → obs: {result.observation[:100]}... done={result.done}")

        trajectory.append({
            "action": response,
            "observation": result.observation,
            "reward": result.reward,
            "done": result.done,
        })

        if result.done:
            return result.reward, trajectory

        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"API Response:\n{result.observation}"})

    # Ran out of steps
    return 0.0, trajectory


# ===== Demo =====
if __name__ == "__main__":
    from task_generator import generate_task

    task = generate_task(difficulty=2)
    print(f"Task: {task.instruction}")
    print(f"Expected: {len(task.expected_actions)} steps")
    print(f"Type: {task.task_type}")

    # Simulate a perfect agent that copies expected actions
    env = APIEnvironment(task)
    obs = env.reset()
    print(f"\nInitial: {obs[:100]}...")

    for i, exp in enumerate(task.expected_actions):
        # Build perfect action string
        action = f"ACTION: {exp['method']} {exp['path']}"
        if exp.get("body"):
            # Replace template vars with real IDs
            body = dict(exp["body"])
            for key, val in env.order_ids.items():
                if "order_id" in body:
                    body["order_id"] = val
            action += f"\nBODY: {json.dumps(body)}"

        print(f"\nStep {i+1}: {action}")
        result = env.step(action)
        print(f"  → {result.observation[:120]}...")
        print(f"  → done={result.done}, reward={result.reward}")

        if result.done:
            print(f"\nFinal: success={result.success}, reward={result.reward}")
            break
