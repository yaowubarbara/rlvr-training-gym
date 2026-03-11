#!/bin/bash
# Phase 1a: Expanded evaluation for v2a and v2c (100 tasks/diff × 3 seeds)
# Only the two from-scratch models that form the core comparison
# Usage: nohup bash /tmp/expanded_eval.sh > /tmp/expanded_eval.log 2>&1 &

set -e
PYTHON=/usr/bin/python3
EVAL_SCRIPT=/tmp/unified_eval.py
SEEDS=(42 123 777)
TASKS_PER_DIFF=100

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

# Models: only v2a and v2c (the core from-scratch comparison)
declare -A MODELS
MODELS[v2a]="/tmp/rlvr_gym/final_model_v2a"
MODELS[v2c]="/tmp/rlvr_gym/final_model_v2c"

echo "============================================================"
echo "  Phase 1a: Expanded Eval (100 tasks/diff)"
echo "  Models: v2a (binary, scratch), v2c (partial, scratch)"
echo "  Seeds: ${SEEDS[*]}"
echo "  Tasks per difficulty: $TASKS_PER_DIFF"
echo "============================================================"

for model_name in v2a v2c; do
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
        echo "--- $model_name seed=$seed (100 tasks/diff) ---"
        outfile="/tmp/rlvr_gym/eval_expanded_${model_name}_seed${seed}.json"

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
echo "  EXPANDED EVAL SUMMARY"
echo "============================================================"

$PYTHON -c "
import json, glob, os
import numpy as np

results = {}
for f in sorted(glob.glob('/tmp/rlvr_gym/eval_expanded_v*_seed*.json')):
    name = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    m = data['metrics']
    if 'v2a' in name:
        model = 'v2a'
    elif 'v2c' in name:
        model = 'v2c'
    else:
        model = name
    if model not in results:
        results[model] = []
    results[model].append(m)

print()
print('='*70)
print('  EXPANDED EVAL: v2a (binary) vs v2c (partial)')
print('  100 tasks/difficulty × 3 seeds = 300 tasks per model per seed')
print('='*70)

for model in ['v2a', 'v2c']:
    if model not in results:
        continue
    runs = results[model]
    n = len(runs)
    print(f'\n  {model} ({n} seeds, 300 tasks each):')

    for key in ['overall_success', 'd1_success', 'd2_success', 'd3_success',
                'avg_partial_progress', 'd3_avg_steps_completed',
                'early_stop_rate', 'd3_early_stop_rate',
                'type_a_rate', 'type_b_rate', 'd3_type_a_rate', 'd3_type_b_rate']:
        vals = [r.get(key, 0) for r in runs]
        avg = np.mean(vals)
        std = np.std(vals)
        mn, mx = min(vals), max(vals)
        if 'rate' in key or 'success' in key or 'progress' in key:
            print(f'    {key}: {avg:.1%} +/- {std:.1%} (range: {mn:.1%} - {mx:.1%})')
        else:
            print(f'    {key}: {avg:.2f} +/- {std:.2f} (range: {mn:.2f} - {mx:.2f})')

    # D3 failure taxonomy
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
echo "[OK] Expanded eval complete"
echo "Timestamp: $(date)"
