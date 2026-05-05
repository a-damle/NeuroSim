"""
exp0_one_job.py — Run a single experiment 0 job.

Usage:
    python exp0_one_job.py bl_1              # FP64 baseline for type 1
    python exp0_one_job.py 3_10_12           # Type 3, w=10, i=12
"""

import sys
import torch
from solver import MixedPrecisionSolver
from acim_matmul import acim_matmul
from test_matrices import TestMatrixGenerator

MATRIX_SIZE = 500
COND = 1e4
MAX_OUTER = 200
TOL = 1e-12
SEED = 42
PRECOND = {1: 'diag', 2: 'diag', 3: 'diag', 4: None, 5: 'diag', 6: None}
INNER_M = {1: 30, 2: 30, 3: 30, 4: 500, 5: 30, 6: 500}

tag = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = TestMatrixGenerator(MATRIX_SIZE, seed=SEED, device=device)
torch.manual_seed(SEED + 1)
b = torch.randn(MATRIX_SIZE, dtype=torch.float64, device=device)


def get_matrix(mtype):
    return {1: gen.type1, 2: lambda: gen.type2(COND), 3: lambda: gen.type3(COND),
            4: lambda: gen.type4(COND), 5: lambda: gen.type5(COND),
            6: lambda: gen.type6(COND)}[mtype]()


if tag.startswith('bl_'):
    mtype = int(tag.split('_')[1])
    A = get_matrix(mtype)
    pc = PRECOND[mtype]
    m = INNER_M[mtype]
    x_ref = torch.linalg.solve(A, b)

    solver = MixedPrecisionSolver(A, b, precond=pc)
    x, hist = solver.solve(max_outer=MAX_OUTER, inner_iters=m, tol=TOL, verbose=False)

    torch.save({
        'type': 'baseline', 'mtype': mtype,
        'x_ref': x_ref, 'history': hist, 'iters': len(hist), 'A': A,
    }, f'results/exp0_{tag}.pt')

else:
    parts = tag.split('_')
    mtype, wb, ib = int(parts[0]), int(parts[1]), int(parts[2])

    A = get_matrix(mtype)
    pc = PRECOND[mtype]
    m = INNER_M[mtype]
    x_ref = torch.linalg.solve(A, b)

    def matmul_fn(A, x):
        return acim_matmul(A, x, weight_bits=wb, input_bits=ib)

    solver = MixedPrecisionSolver(A, b, matmul_fn=matmul_fn, precond=pc)
    x, hist = solver.solve(max_outer=MAX_OUTER, inner_iters=m, tol=TOL, verbose=False)

    rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
    converged = hist[-1] < TOL * 10

    torch.save({
        'type': 'quantized', 'mtype': mtype, 'wb': wb, 'ib': ib,
        'history': hist, 'iters': len(hist),
        'final_residual': hist[-1], 'rel_error': rel_err, 'converged': converged,
    }, f'results/exp0_{tag}.pt')
