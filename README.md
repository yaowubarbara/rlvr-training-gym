# Naive Stepwise Partial Rewards Induce Early-Stop Collapse in GRPO Training of Sequential API Agents

This repository contains the code, training scripts, evaluation harness, and figures for the paper.

## Key Finding

Training a 0.5B LLM agent with GRPO on a deterministic API workflow gym, we compare outcome-dominant (binary) rewards against naive stepwise partial rewards. The partial reward variant (v2c) scores 14% overall vs. 79% for binary (v2a), a 5.5x gap that widens to 6.7x on 900 expanded tasks. The root cause is **early-stop collapse**: the agent learns to execute one correct step and stop, collecting partial credit without attempting harder subsequent actions. On D3 (hard) tasks, v2c exhibits 100% Type B (early stop) failure across all seeds.

## Repository Structure

```
gym/            API training gym (order management system clone)
  api_system.py       Simulated order management API with state machine
  task_generator.py   Procedural task generator (infinite variations)
  verifier.py         Deterministic binary verifier (RLVR core)
  env.py              Multi-turn agent loop environment

training/       GRPO training scripts for each variant
  train_grpo.py       Base GRPO training loop
  train_v2a.py        Outcome-dominant binary reward (79%)
  train_v2c.py        Naive stepwise partial reward (14%)
  train_v2b.py        Binary + curriculum (76%)
  train_v3a_sft.py    SFT baseline (46%)
  train_v4_combined.py  Partial reward continuation from v2b (82%)
  train_v5.py         Milestone hybrid (0%)

eval/           Evaluation harness
  unified_eval.py     Unified eval with failure taxonomy
  multi_seed_eval.sh  Multi-seed evaluation runner (3 seeds)
  expanded_eval.sh    900-task expanded evaluation

figures/        Paper figures (Nature/Cell palette)
  make_figures.py     Figure generation script
  fig1_overall.png    Training variant comparison (7 variants, 3 seeds)
  fig2_expanded.png   v2a vs v2c by difficulty (900 tasks)
  fig3_taxonomy.png   D3 failure taxonomy (Type A / Type B / Other)

paper/
  paper.tex           LaTeX source
```

## The Gym

The gym simulates an e-commerce order management API with a state machine (created, paid, shipped, delivered, refunded, cancelled). Tasks are procedurally generated with three difficulty levels based on the number of sequential API calls required.

The verifier checks each API call against the known correct sequence and provides binary (pass/fail) or partial (steps_completed / steps_total) rewards. No learned reward model is needed since correctness is fully determined by the API state machine.

## Training Variants

| Variant | Reward | Init | Overall | D3 Type B |
|---------|--------|------|---------|-----------|
| v2a | outcome-dominant binary | scratch | 79 +/- 3% | 27% |
| v2c | naive stepwise partial | scratch | 14 +/- 3% | 100% |
| v2b | binary + curriculum | scratch | 76 +/- 4% | 0% |
| v4 | partial (cont. from v2b) | cont. | 82 +/- 2% | 27% |
| v5 | milestone hybrid | scratch | 0% | 33% |
| v1 | format-only binary | scratch | 63% | n/a |
| v3a | SFT | scratch | 46% | n/a |

## Reproducing

1. Start the API gym

```bash
cd gym && uvicorn api_system:app --port 8000
```

2. Run training (requires GPU + transformers + trl)

```bash
python training/train_v2a.py   # binary reward
python training/train_v2c.py   # partial reward
```

3. Evaluate

```bash
python eval/unified_eval.py --model /path/to/checkpoint --seed 42 --tasks-per-diff 10
```

4. Generate figures

```bash
python figures/make_figures.py
```

## Requirements

Python 3.10+, PyTorch, transformers, trl, FastAPI, uvicorn, httpx, matplotlib, numpy

## License

MIT
