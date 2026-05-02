"""
run_validation.py — Verify solver correctness before running experiments.

Run:  python run_validation.py
"""

import torch
from solver import MixedPrecisionSolver
from acim_matmul import acim_matmul
from test_matrices import TestMatrixGenerator


PRECOND_MAP = {1: 'diag', 2: 'diag', 3: 'diag', 4: None, 5: 'diag', 6: None}
INNER_ITERS_MAP = {1: 50, 2: 50, 3: 50, 4: 100, 5: 50, 6: 100}
TYPE_NAMES = {
    1: "Diag dominant", 2: "Log-uniform σ, +λ",
    3: "Clustered σ, +λ", 4: "Clustered σ",
    5: "Arithmetic σ, +λ", 6: "Arithmetic σ",
}

N = 100
COND = 1e4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_quantized_matmul(wb, ib):
    def fn(A, x):
        return acim_matmul(A, x, weight_bits=wb, input_bits=ib)
    return fn


def test_fp64_all_types():
    """FP64 baseline on all 6 matrix types."""
    print("=" * 70)
    print(f"FP64 baseline (n={N}, cond={COND:.0e})")
    print("=" * 70)

    gen = TestMatrixGenerator(N, seed=42, device=DEVICE)
    torch.manual_seed(99)
    b = torch.randn(N, dtype=torch.float64, device=DEVICE)

    all_pass = True
    for mtype in sorted(TYPE_NAMES.keys()):
        fn = {1: gen.type1, 2: lambda: gen.type2(COND), 3: lambda: gen.type3(COND),
              4: lambda: gen.type4(COND), 5: lambda: gen.type5(COND), 6: lambda: gen.type6(COND)}
        A = fn[mtype]()
        pc = PRECOND_MAP[mtype]
        m = INNER_ITERS_MAP[mtype]

        solver = MixedPrecisionSolver(A, b, precond=pc)
        x, hist = solver.solve(max_outer=200, inner_iters=m, tol=1e-10, verbose=False)

        x_ref = torch.linalg.solve(A, b)
        rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
        ok = hist[-1] < 1e-8

        pc_str = "diag" if pc else "none"
        print(f"  Type {mtype} ({TYPE_NAMES[mtype]:22s}) precond={pc_str:4s} m={m:3d}  "
              f"iters={len(hist):3d}  ||r||={hist[-1]:.2e}  rel_err={rel_err:.2e}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}\n")
    return all_pass


def test_quantized_all_types():
    """Quantized convergence check across all 6 matrix types."""
    print("=" * 70)
    print(f"Quantized convergence check — all types (n={N}, cond={COND:.0e})")
    print("=" * 70)

    gen = TestMatrixGenerator(N, seed=42, device=DEVICE)
    torch.manual_seed(99)
    b = torch.randn(N, dtype=torch.float64, device=DEVICE)

    for mtype in sorted(TYPE_NAMES.keys()):
        fn = {1: gen.type1, 2: lambda: gen.type2(COND), 3: lambda: gen.type3(COND),
              4: lambda: gen.type4(COND), 5: lambda: gen.type5(COND), 6: lambda: gen.type6(COND)}
        A = fn[mtype]()
        x_ref = torch.linalg.solve(A, b)
        pc = PRECOND_MAP[mtype]
        m = INNER_ITERS_MAP[mtype]

        pc_str = "diag" if pc else "none"
        print(f"\n  Type {mtype}: {TYPE_NAMES[mtype]}  (precond={pc_str}, m={m})")
        print(f"  {'bits':>6s}  {'iters':>6s}  {'||r||':>12s}  {'rel_err':>12s}  status")
        print(f"  {'-'*56}")

        for bits in [4, 6, 8, 10, 12, 16]:
            matmul_fn = make_quantized_matmul(bits, bits)
            solver = MixedPrecisionSolver(A, b, matmul_fn=matmul_fn, precond=pc)
            x, hist = solver.solve(max_outer=100, inner_iters=m, tol=1e-10, verbose=False)

            rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
            conv = hist[-1] < 1e-8

            print(f"  {bits:6d}  {len(hist):6d}  {hist[-1]:12.2e}  {rel_err:12.2e}  "
                  f"{'CONV ✓' if conv else 'NO CONV'}")

    print()


if __name__ == "__main__":
    t1 = test_fp64_all_types()
    test_quantized_all_types()

    print("=" * 70)
    if t1:
        print("FP64 baselines pass. Ready for experiment 0.")
        print("(Non-convergence at low precision is expected — that's the experiment.)")
    else:
        print("FP64 baselines failed. Debug before running experiments.")
    print("=" * 70)
