"""
experiment_0.py — Integer quantization viability study.

Sweeps weight_bits and input_bits across symmetric matrix types (1, 2, 3, 5)
where the diagonal preconditioner is effective. Types 4, 6 (non-symmetric)
are included as FP64 baselines only — they require m=n inner iterations
without preconditioning, which eliminates the CIM efficiency advantage.

Outputs:
  - results/exp0_results.pt      (all convergence data)
  - results/exp0_convergence.png (convergence curves)

Run:  python experiment_0.py
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

from solver import MixedPrecisionSolver
from acim_matmul import acim_matmul
from test_matrices import TestMatrixGenerator


# ── Configuration ─────────────────────────────────────────────────

MATRIX_SIZE    = 500
COND_NUMBER    = 1e4
MAX_OUTER      = 200
TOL            = 1e-12
SEED           = 42

WEIGHT_BITS = [4, 6, 8, 10, 12, 16]
INPUT_BITS  = [4, 6, 8, 10, 12, 16]

# All types get FP64 baselines; only symmetric types get quantized sweep
ALL_TYPES   = [1, 2, 3, 4, 5, 6]
SWEEP_TYPES = [1, 2, 3, 5]

PRECOND_MAP     = {1: 'diag', 2: 'diag', 3: 'diag', 4: None, 5: 'diag', 6: None}
INNER_ITERS_MAP = {1: 30, 2: 30, 3: 30, 4: 500, 5: 30, 6: 500}

TYPE_NAMES = {
    1: "Diag dominant", 2: "Log-uniform σ, +λ",
    3: "Clustered σ, +λ", 4: "Clustered σ",
    5: "Arithmetic σ, +λ", 6: "Arithmetic σ",
}

OUTPUT_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────

def get_matrix(gen, mtype):
    if mtype == 1:
        return gen.type1()
    return {2: gen.type2, 3: gen.type3, 4: gen.type4,
            5: gen.type5, 6: gen.type6}[mtype](cond=COND_NUMBER)


def make_matmul_fn(wb, ib):
    def fn(A, x):
        return acim_matmul(A, x, weight_bits=wb, input_bits=ib)
    return fn


# ── Main ──────────────────────────────────────────────────────────

def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gen = TestMatrixGenerator(MATRIX_SIZE, seed=SEED, device=DEVICE)
    torch.manual_seed(SEED + 1)
    b = torch.randn(MATRIX_SIZE, dtype=torch.float64, device=DEVICE)

    results = {}
    baselines = {}

    # ── FP64 baselines (all types) ────────────────────────────────
    print("Running FP64 baselines...\n")
    for mtype in ALL_TYPES:
        A = get_matrix(gen, mtype)
        x_ref = torch.linalg.solve(A, b)
        pc = PRECOND_MAP[mtype]
        m = INNER_ITERS_MAP[mtype]

        solver = MixedPrecisionSolver(A, b, precond=pc)
        x, hist = solver.solve(max_outer=MAX_OUTER, inner_iters=m,
                                tol=TOL, verbose=False)

        baselines[mtype] = {"x_ref": x_ref, "history": hist,
                            "iters": len(hist), "A": A}

        pc_str = "diag" if pc else "none"
        conv = "✓" if hist[-1] < TOL * 10 else "✗"
        sweep_note = "" if mtype in SWEEP_TYPES else "  (baseline only, m=n)"
        print(f"  Type {mtype} ({TYPE_NAMES[mtype]:22s}) precond={pc_str:4s} m={m:3d}  "
              f"{len(hist):3d} iters  ||r||={hist[-1]:.2e}  {conv}{sweep_note}")

    # ── Quantization sweep (symmetric types only) ─────────────────
    total = len(WEIGHT_BITS) * len(INPUT_BITS) * len(SWEEP_TYPES)
    print(f"\nRunning quantized sweep on types {SWEEP_TYPES}: {total} configurations...\n")

    count = 0
    for mtype in SWEEP_TYPES:
        A = baselines[mtype]["A"]
        x_ref = baselines[mtype]["x_ref"]
        pc = PRECOND_MAP[mtype]
        m = INNER_ITERS_MAP[mtype]

        for wb, ib in product(WEIGHT_BITS, INPUT_BITS):
            count += 1

            solver = MixedPrecisionSolver(A, b, matmul_fn=make_matmul_fn(wb, ib),
                                           precond=pc)
            x, hist = solver.solve(max_outer=MAX_OUTER, inner_iters=m,
                                    tol=TOL, verbose=False)

            rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
            converged = hist[-1] < TOL * 10

            results[(mtype, wb, ib)] = {
                "history": hist, "iters": len(hist),
                "final_residual": hist[-1], "rel_error": rel_err,
                "converged": converged,
            }

            if count % 24 == 0 or count == total:
                print(f"  [{count:3d}/{total}] Type {mtype}, w={wb:2d}, i={ib:2d}: "
                      f"{len(hist):3d} iters  ||r||={hist[-1]:.2e}  "
                      f"{'✓' if converged else '✗'}")

    # ── Save ──────────────────────────────────────────────────────
    torch.save({"results": results, "baselines": baselines,
                "config": {"matrix_size": MATRIX_SIZE, "cond": COND_NUMBER,
                           "max_outer": MAX_OUTER, "inner_iters_map": INNER_ITERS_MAP,
                           "tol": TOL, "weight_bits": WEIGHT_BITS,
                           "input_bits": INPUT_BITS, "precond_map": PRECOND_MAP,
                           "sweep_types": SWEEP_TYPES, "all_types": ALL_TYPES}},
               os.path.join(OUTPUT_DIR, "exp0_results.pt"))
    print(f"\nResults saved to {OUTPUT_DIR}/exp0_results.pt")

    print_summary(results, baselines)
    plot_convergence(results, baselines)


def print_summary(results, baselines):
    print("\n" + "=" * 80)
    print("EXPERIMENT 0: Iterations to convergence (- = did not converge)")
    print("=" * 80)

    for mtype in ALL_TYPES:
        pc_str = "diag" if PRECOND_MAP[mtype] else "none"
        m = INNER_ITERS_MAP[mtype]
        bl = baselines[mtype]
        bl_conv = bl['history'][-1] < TOL * 10
        bl_status = f"{bl['iters']} iters" if bl_conv else "DID NOT CONVERGE"

        print(f"\n  Type {mtype}: {TYPE_NAMES[mtype]}  (precond={pc_str}, m={m})")
        print(f"  FP64 baseline: {bl_status}")

        if mtype not in SWEEP_TYPES:
            print(f"  (Skipped quantized sweep — requires m=n, no CIM advantage)")
            continue

        print()
        header = "  w\\i  " + "".join(f"{ib:>6d}" for ib in INPUT_BITS)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for wb in WEIGHT_BITS:
            row = f"  {wb:4d}  "
            for ib in INPUT_BITS:
                r = results[(mtype, wb, ib)]
                row += f"{r['iters']:6d}" if r["converged"] else f"{'  -':>6s}"
            print(row)


def plot_convergence(results, baselines):
    sym_bits = [4, 6, 8, 10, 16]
    colors = {4: 'red', 6: 'orange', 8: 'blue', 10: 'green', 16: 'purple'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, mtype in enumerate(sorted(TYPE_NAMES.keys())):
        ax = axes[idx // 3][idx % 3]

        h = baselines[mtype]["history"]
        ax.semilogy(range(len(h)), h, 'k--', lw=2, label='FP64')

        if mtype in SWEEP_TYPES:
            for bits in sym_bits:
                key = (mtype, bits, bits)
                if key in results:
                    h = results[key]["history"]
                    ax.semilogy(range(len(h)), h, color=colors[bits], lw=1.5,
                               label=f'{bits}-bit')

        pc_str = "diag" if PRECOND_MAP[mtype] else "none"
        note = "" if mtype in SWEEP_TYPES else "\n(no quant sweep — needs m=n)"
        ax.set_title(f'Type {mtype}: {TYPE_NAMES[mtype]}\n(precond={pc_str}){note}',
                     fontsize=9)
        ax.set_xlabel('Outer iteration')
        ax.set_ylabel('||r||₂')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp0_convergence.png")
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    run_experiment()
