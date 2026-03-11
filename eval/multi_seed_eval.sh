#!/bin/bash
# Multi-seed eval runner for ALL key variants
# Runs unified_eval.py with 3 seeds per model, 10 tasks per difficulty
# Usage: nohup bash /tmp/multi_seed_eval.sh > /tmp/multi_seed_eval.log 2>&1 &

set -e
PYTHON=/usr/bin/python3
EVAL_SCRIPT=/tmp/unified_eval.py
SEEDS=(42 123 777)
TASKS_PER_DIFF=10

# Start API server
cd /tmp/rlvr_gym
$PYTHON -m uvicorn api_system:app --port 8000 --host 127.0.0.1 --log-level warning &
API_PID=$!
sleep 5
echo "[*] API server started (PID: $API_PID)"

cleanup() {
    kill $API_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    echo "[*] API server stopped"
}
trap cleanup EXIT

# Models to evaluate — priority order (most important first)
declare -A MODELS
MODELS[v2a]="/tmp/rlvr_gym/final_model_v2a"
MODELS[v4_fixed]="/tmp/rlvr_gym/final_model_v4"
MODELS[v2b_fixed]="/tmp/rlvr_gym/final_model_v2b"
MODELS[v2c]="/tmp/rlvr_gym/final_model_v2c"

echo "============================================================"
echo "  Multi-Seed Unified Eval"
echo "  Models: v2a (binary), v4 (partial), v2b (binary+curric), v2c (partial-loop)"
echo "  Seeds: ${SEEDS[*]}"
echo "  Tasks per difficulty: $TASKS_PER_DIFF"
echo "============================================================"

# Run in priority order: v2a first (star), v4 second (anti-star), then others
for model_name in v2a v4_fixed v2b_fixed v2c; do
    model_path=${MODELS[$model_name]}

    if [ ! -f "$model_path/model.safetensors" ]; then
        echo "[SKIP] $model_name — model not found at $model_path"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Evaluating: $model_name ($model_path)"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "--- $model_name seed=$seed ---"
        outfile="/tmp/rlvr_gym/eval_${model_name}_seed${seed}.json"

        if [ -f "$outfile" ]; then
            echo "[SKIP] Already exists: $outfile"
            continue
        fi

        $PYTHON $EVAL_SCRIPT \
            --model "$model_path" \
            --seed "$seed" \
            --tasks-per-diff "$TASKS_PER_DIFF" \
            --output "$outfile"

        echo "[OK] $model_name seed=$seed done"
        echo ""
    done
done

# === Aggregate summary ===
echo ""
echo "============================================================"
echo "  AGGREGATE SUMMARY"
echo "============================================================"

$PYTHON -c "
import json, glob, os
import numpy as np

results = {}
for f in sorted(glob.glob('/tmp/rlvr_gym/eval_v*_seed*.json')):
    name = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    m = data['metrics']
    # Extract model name
    if 'v2a' in name:
        model = 'v2a'
    elif 'v2b' in name:
        model = 'v2b'
    elif 'v2c' in name:
        model = 'v2c'
    elif 'v4' in name:
        model = 'v4'
    else:
        model = name.split('_seed')[0]
    if model not in results:
        results[model] = []
    results[model].append(m)

print()
print('='*70)
print('  MODEL COMPARISON (multi-seed averages)')
print('='*70)

# Header
print(f\"{'Model':<8} {'Overall':>8} {'D1':>8} {'D2':>8} {'D3':>8} {'D3_TypeA':>10} {'D3_TypeB':>10}\")
print('-'*64)

for model in ['v2a', 'v2b', 'v2c', 'v4']:
    if model not in results:
        continue
    runs = results[model]
    n = len(runs)

    overall = np.mean([r['overall_success'] for r in runs])
    d1 = np.mean([r['d1_success'] for r in runs])
    d2 = np.mean([r['d2_success'] for r in runs])
    d3 = np.mean([r['d3_success'] for r in runs])
    d3_ta = np.mean([r.get('d3_type_a_rate', 0) for r in runs])
    d3_tb = np.mean([r.get('d3_type_b_rate', 0) for r in runs])

    print(f'{model} ({n}s) {overall:>7.1%} {d1:>7.1%} {d2:>7.1%} {d3:>7.1%} {d3_ta:>9.1%} {d3_tb:>9.1%}')

print()
print('='*70)
print('  DETAILED PER-MODEL STATS')
print('='*70)

for model in ['v2a', 'v2b', 'v2c', 'v4']:
    if model not in results:
        continue
    runs = results[model]
    n = len(runs)
    print(f'\n  {model} ({n} seeds):')

    for key in ['overall_success', 'd1_success', 'd2_success', 'd3_success',
                'avg_partial_progress', 'd3_avg_steps_completed',
                'final_state_fail_rate', 'early_stop_rate', 'd3_early_stop_rate',
                'type_a_rate', 'type_b_rate', 'd3_type_a_rate', 'd3_type_b_rate']:
        vals = [r.get(key, 0) for r in runs]
        avg = np.mean(vals)
        std = np.std(vals)
        mn, mx = min(vals), max(vals)
        if 'rate' in key or 'success' in key or 'progress' in key:
            print(f'    {key}: {avg:.1%} ± {std:.1%} (range: {mn:.1%}–{mx:.1%})')
        else:
            print(f'    {key}: {avg:.2f} ± {std:.2f} (range: {mn:.2f}–{mx:.2f})')

    # Aggregate failure taxonomy
    all_tax = {}
    for r in runs:
        for mode, cnt in r.get('d3_failure_taxonomy', {}).items():
            all_tax[mode] = all_tax.get(mode, 0) + cnt
    total_d3_fail = sum(r.get('d3_total_failures', 0) for r in runs)
    if all_tax:
        print(f'    D3 failure taxonomy (across all seeds, {total_d3_fail} total):')
        for mode, cnt in sorted(all_tax.items(), key=lambda x: -x[1]):
            print(f'      {mode}: {cnt} ({cnt/max(total_d3_fail,1):.0%})')
"

echo ""
echo "[OK] Multi-seed eval complete"
echo "Timestamp: $(date)"
