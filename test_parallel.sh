#!/bin/bash
# test_parallel.sh — Test the parallel pipeline before full sweep.
#
# Runs 12 jobs (4 types × 3 K) with minimal configs.
# NO AUTO CLEANUP — check results manually, then delete when satisfied.
#
# Usage:
#   bash test_parallel.sh rram    # test with RRAM
#   bash test_parallel.sh sram    # test with SRAM (after Param.cpp change + recompile)

DEVICE=${1:-sram}

echo "=== Running parallel test ($DEVICE, 12 jobs) ==="
echo ""
bash run_sweep.sh $DEVICE 12 sweep_config_parallel_test.json
echo ""

echo "=== Verifying ==="
python -c "
import torch
d = torch.load('results/exp12_${DEVICE}_results.pt', weights_only=False, map_location='cpu')
n = len(d['results'])
nb = len(d.get('baselines', {}))
types = sorted(set(k[0] for k in d['results'].keys()))
Ks = sorted(set(k[4] for k in d['results'].keys()))

print(f'Configs: {n} (expected 12)')
print(f'Baselines: {nb} (expected 4)')
print(f'Types: {types} (expected [1,2,3,5])')
print(f'K: {Ks} (expected [1,4,8])')

ok = n == 12 and nb == 4 and types == [1,2,3,5] and Ks == [1,4,8]
print(f'\nResult: {\"PASS ✓\" if ok else \"FAIL ✗\"}\n')
"

echo "=== Files created (NOT auto-deleted) ==="
echo "  results/exp12_${DEVICE}_results.pt          (merged)"
echo "  results/exp12_${DEVICE}_t*_results.pt       (per-job)"
echo "  results/log_${DEVICE}_t*.txt                (logs)"
echo "  jobs/                                        (job configs)"
echo ""
echo "When satisfied, clean up manually:"
echo "  rm -f results/exp12_${DEVICE}_t*_results.pt"
echo "  rm -f results/log_${DEVICE}_t*.txt"
echo "  rm -rf jobs/"
echo "  rm -rf layer_record_*/"
