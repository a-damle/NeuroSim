#!/bin/bash
# run_sweep.sh — Run experiment sweep with perfect load balancing.
#
# NO AUTO CLEANUP — all job configs, logs, and intermediate results
# are preserved. Clean up manually with cleanup.sh when ready.
#
# Prerequisites for SRAM:
#   1. Edit NeuroSIM/Param.cpp: memcelltype = 1
#   2. cd NeuroSIM && make && cd ..
#
# Usage:
#   bash run_sweep.sh rram              # 12 parallel (default)
#   bash run_sweep.sh rram 8            # 8 parallel
#   bash run_sweep.sh sram 12           # SRAM
#   bash run_sweep.sh rram 12 my.json   # custom config

DEVICE=${1:-rram}
NPROC=${2:-24}
CONFIG=${3:-sweep_config.json}

echo "=== Generating job configs ==="
python split_config.py $CONFIG

NJOBS=$(wc -l < jobs/job_list.txt)
echo ""
echo "=== Starting sweep: $NJOBS jobs, $NPROC parallel (device=$DEVICE) ==="
echo ""
mkdir -p results

# Run one job — called by xargs
run_one() {
    TAG=$1
    CFG="jobs/job_${TAG}.json"
    LOG="results/log_${DEVICE}_${TAG}.txt"
    python experiment_1_2.py --device $DEVICE --config $CFG --suffix _${TAG} > $LOG 2>&1
    
    # Atomic counter increment
    flock results/.counter.lock bash -c 'N=$(cat results/.counter 2>/dev/null || echo 0); echo $((N+1)) > results/.counter; cat results/.counter'
    DONE=$(cat results/.counter)

    PT="results/exp12_${DEVICE}_${TAG}_results.pt"
    if [ -f "$PT" ]; then
        echo "  [$DONE/$NJOBS] ✓ $TAG"
    else
        echo "  [$DONE/$NJOBS] ✗ $TAG (check $LOG)"
    fi
}
export -f run_one
export DEVICE
export NJOBS

# Reset counter
echo 0 > results/.counter
touch results/.counter.lock

cat jobs/job_list.txt | xargs -P $NPROC -I {} bash -c 'run_one "{}"'

echo ""
echo "=== All jobs complete ==="
echo ""

# Check for failures
FAILED=0
PASSED=0
for TAG in $(cat jobs/job_list.txt); do
    PT="results/exp12_${DEVICE}_${TAG}_results.pt"
    if [ -f "$PT" ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "  FAILED: $TAG"
    fi
done
echo "Passed: $PASSED / $NJOBS"
if [ $FAILED -gt 0 ]; then
    echo "$FAILED jobs failed. Check results/log_${DEVICE}_*.txt"
fi
echo ""

# Merge all results
echo "=== Merging results ==="
python merge_results.py \
    results/exp12_${DEVICE}_t*_results.pt \
    -o results/exp12_${DEVICE}_results.pt
echo ""

# Summary
echo "=== Summary ==="
python -c "
import torch
d = torch.load('results/exp12_${DEVICE}_results.pt', weights_only=False, map_location='cpu')
n = len(d['results'])
nb = len(d.get('baselines', {}))
types = sorted(set(k[0] for k in d['results'].keys()))
Ks = sorted(set(k[4] for k in d['results'].keys()))
n_conv = sum(1 for r in d['results'].values() if r.get('converged'))
n_div = sum(1 for r in d['results'].values() if r.get('stop_reason') == 'diverged')
n_max = sum(1 for r in d['results'].values() if r.get('stop_reason') == 'max_iters')
n_skip = sum(1 for r in d['results'].values() if r.get('stop_reason') == 'skipped')
print(f'Total: {n} configs, {nb} baselines')
print(f'Types: {types}, K: {Ks}')
print(f'Converged: {n_conv}, Diverged: {n_div}, Max iters: {n_max}, Skipped: {n_skip}')
"
echo ""
echo "=== Done ==="
echo "Results: results/exp12_${DEVICE}_results.pt"
echo "Per-job results: results/exp12_${DEVICE}_t*_results.pt"
echo "Logs: results/log_${DEVICE}_*.txt"
echo "Job configs: jobs/"

# Clean counter files
rm -f results/.counter results/.counter.lock
