# RLVR Training Gym

A mock enterprise API environment for training LLM agents with **Reinforcement Learning from Verifiable Rewards (RLVR)**.

Built as a functional training substrate: a state-machine-verified order management API, Boltzmann adaptive curriculum, and GRPO training loop. The environment generates tasks of varying difficulty, the agent produces API call sequences, and a verifier checks correctness against ground truth — no human judge required.

## Results

Trained Qwen2.5-1.5B with GRPO across 6 configurations:

| Variant | Reward Design | Easy | Medium | Hard | **Total** |
|---------|--------------|------|--------|------|-----------|
| v1 (baseline) | format only | 10/10 | 9/10 | 0/10 | 19/30 = 63% |
| **v2a** | **format + binary accuracy** | **9/10** | **10/10** | **7/10** | **26/30 = 86%** |
| v2b | v2a + Boltzmann curriculum | 10/10 | 9/10 | 0/10 | 19/30 = 63% |
| v2c | format + step-by-step partial | 8/10 | 0/10 | 0/10 | 8/30 = 26% |
| v3a (SFT) | supervised fine-tuning | 10/10 | 4/10 | 0/10 | 14/30 = 46% |
| v4 | format + stepwise execution | 8/10 | 3/10 | 0/10 | 11/30 = 36% |

**Key finding:** Binary verifiable rewards (v2a, 86%) dramatically outperform partial/step-based rewards (v2c, 26%). Partial credit creates reward hacking — the model learns to do step 1 and stop.

## Architecture

```
Task Generator ──→ Agent (LLM) ──→ Verifier
      │                                  │
      │         Mock API Server          │
      │         (FastAPI + State Machine) │
      └──────────────────────────────────┘
                    ↓
              GRPO Reward Signal
```

### Components

- **`api_system.py`** — FastAPI mock order management API with state machine (pending → paid → shipped → delivered, with cancel/refund branches)
- **`task_generator.py`** — Generates tasks across 5 types and 3 difficulty levels, with Boltzmann adaptive curriculum
- **`verifier.py`** — Verifies agent actions against expected sequences, handling dynamic IDs and state dependencies
- **`env.py`** — Gymnasium-compatible environment wrapper for step-by-step execution
- **`train_grpo.py`** — GRPO training loop with configurable reward functions

### Task Types

| Difficulty | Task Types |
|-----------|-----------|
| Easy (d=1) | `create_order`, `create_and_check`, `create_and_cancel` |
| Medium (d=2) | `create_and_pay`, `create_pay_ship` |
| Hard (d=3) | `full_delivery`, `full_lifecycle_with_refund` |

### Reward Functions

1. **`format_reward`** — Checks output format (ACTION/BODY structure)
2. **`accuracy_reward`** — Binary: 1.0 if all steps correct, 0.0 otherwise
3. **`step_reward`** — Partial credit: steps_completed / steps_total (causes reward hacking!)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the mock API
uvicorn api_system:app --port 8000

# Run tests
python test_verifier.py

# Train (requires GPU)
python train_grpo.py
```

## Blog Post

Full writeup with analysis: [Building a Zendesk Training Gym for LLM Agents](https://yaowu-portfolio.vercel.app/blog/building-zendesk-training-gym-llm-agents)

## References

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv:2501.12948)
- WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL (arXiv:2411.02337)
- DAPO: An Open-Source LLM RL System at Scale (arXiv:2503.14476)
- TinyV: Rethinking Verifier Design for Scalable RLVR (arXiv:2505.14625)

## License

MIT
