#!/bin/bash
# run_exp0_parallel.sh — Run experiment 0 with per-config parallelism.
#
# 222 jobs (6 baselines + 216 quantized), 12 parallel via xargs.
#
# Usage:  bash run_exp0_parallel.sh [nproc]

NPROC=${1:-12}
mkdir -p results

# Generate job list
python -c "
types = [1,2,3,4,5,6]
bits = [4,6,8,10,12,16]
jobs = []
for t in types:
    jobs.append(f'bl_{t}')
for t in types:
    for wb in bits:
        for ib in bits:
            jobs.append(f'{t}_{wb}_{ib}')
with open('results/exp0_job_list.txt', 'w') as f:
    for j in jobs:
        f.write(j + '\n')
print(f'{len(jobs)} jobs generated')
"

NJOBS=$(wc -l < results/exp0_job_list.txt)
echo "=== Experiment 0: $NJOBS jobs, $NPROC parallel ==="
echo ""

run_one() {
    TAG=$1
    python exp0_one_job.py $TAG 2>&1
    if [ -f "results/exp0_${TAG}.pt" ]; then
        echo "  ✓ $TAG"
    else
        echo "  ✗ $TAG"
    fi
}
export -f run_one

cat results/exp0_job_list.txt | xargs -P $NPROC -I {} bash -c 'run_one "{}"'

echo ""
echo "=== Merging ==="

python -c "
import torch, glob

all_results = {}
all_baselines = {}

for f in sorted(glob.glob('results/exp0_*.pt')):
    if 'results' in f or 'job_list' in f:
        continue
    d = torch.load(f, weights_only=False, map_location='cpu')

    if d['type'] == 'baseline':
        mt = d['mtype']
        all_baselines[mt] = {
            'x_ref': d['x_ref'], 'history': d['history'],
            'iters': d['iters'], 'A': d['A'],
        }
    else:
        key = (d['mtype'], d['wb'], d['ib'])
        all_results[key] = {
            'history': d['history'], 'iters': d['iters'],
            'final_residual': d['final_residual'],
            'rel_error': d['rel_error'], 'converged': d['converged'],
        }

torch.save({
    'results': all_results,
    'baselines': all_baselines,
    'config': {
        'matrix_size': 500, 'cond': 1e4, 'max_outer': 200,
        'inner_iters_map': {1:30, 2:30, 3:30, 4:500, 5:30, 6:500},
        'tol': 1e-12, 'weight_bits': [4,6,8,10,12,16],
        'input_bits': [4,6,8,10,12,16],
        'precond_map': {1:'diag', 2:'diag', 3:'diag', 4:None, 5:'diag', 6:None},
        'sweep_types': [1,2,3,4,5,6], 'all_types': [1,2,3,4,5,6],
    }
}, 'results/exp0_results.pt')

print(f'Total: {len(all_results)} results, {len(all_baselines)} baselines')
print()

TYPE_NAMES = {1:'Diag dominant', 2:'Log-uniform σ', 3:'Clustered σ +λ',
              4:'Clustered σ', 5:'Arithmetic σ +λ', 6:'Arithmetic σ'}
BITS = [4,6,8,10,12,16]

for mtype in [1,2,3,4,5,6]:
    bl = all_baselines.get(mtype)
    if bl:
        print(f'Type {mtype} ({TYPE_NAMES[mtype]}):')
        print(f'  FP64: {bl[\"iters\"]} iters, ||r||={bl[\"history\"][-1]:.2e}')

    entries = {k:v for k,v in all_results.items() if k[0] == mtype}
    n_conv = sum(1 for v in entries.values() if v['converged'])
    print(f'  Quantized: {n_conv}/36 converged')

    header = '  w\\\i  ' + ''.join(f'{ib:>6d}' for ib in BITS)
    print(header)
    for wb in BITS:
        row = f'  {wb:4d}  '
        for ib in BITS:
            r = entries.get((mtype, wb, ib))
            if r and r['converged']:
                row += f'{r[\"iters\"]:6d}'
            else:
                row += '     -'
        print(row)
    print()
"

echo "=== Done ==="
echo "Results: results/exp0_results.pt"
echo "Per-job: results/exp0_*.pt"
