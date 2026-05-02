"""
inspect_exp0.py — Load experiment 0 results and show convergence summary.

Run:  python inspect_exp0.py
"""

import torch

data = torch.load("results/exp0_results.pt", weights_only=False)
results = data["results"]
baselines = data["baselines"]
config = data["config"]

TYPE_NAMES = {
    1: "Diag dominant", 2: "Log-uniform σ, +λ",
    3: "Clustered σ, +λ", 4: "Clustered σ",
    5: "Arithmetic σ, +λ", 6: "Arithmetic σ",
}

WEIGHT_BITS = config["weight_bits"]
INPUT_BITS = config["input_bits"]

print("=" * 80)
print("EXPERIMENT 0 RESULTS")
print(f"Matrix size: {config['matrix_size']}, cond: {config['cond']:.0e}, "
      f"tol: {config['tol']:.0e}, max_outer: {config['max_outer']}")
print("=" * 80)

# Show convergence tables
for mtype in sorted(baselines.keys()):
    bl = baselines[mtype]
    bl_conv = bl['history'][-1] < config['tol'] * 10
    print(f"\nType {mtype}: {TYPE_NAMES[mtype]}")
    print(f"  FP64 baseline: {bl['iters']} iters, ||r||={bl['history'][-1]:.2e}  "
          f"{'✓' if bl_conv else '✗'}")

    # Check if this type was in the quantized sweep
    any_results = any((mtype, wb, ib) in results for wb in WEIGHT_BITS for ib in INPUT_BITS)
    if not any_results:
        print("  (No quantized sweep — baseline only)")
        continue

    # Iteration count table
    print(f"\n  Iterations to convergence (- = did not converge):")
    header = "  w\\i  " + "".join(f"{ib:>6d}" for ib in INPUT_BITS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for wb in WEIGHT_BITS:
        row = f"  {wb:4d}  "
        for ib in INPUT_BITS:
            key = (mtype, wb, ib)
            if key in results:
                r = results[key]
                row += f"{r['iters']:6d}" if r["converged"] else f"{'  -':>6s}"
            else:
                row += f"{'  n/a':>6s}"
        print(row)

    # Final residual table
    print(f"\n  Final residual (log10):")
    header = "  w\\i  " + "".join(f"{ib:>8d}" for ib in INPUT_BITS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for wb in WEIGHT_BITS:
        row = f"  {wb:4d}  "
        for ib in INPUT_BITS:
            key = (mtype, wb, ib)
            if key in results:
                r = results[key]
                import math
                log_res = math.log10(r['final_residual']) if r['final_residual'] > 0 else float('-inf')
                row += f"{log_res:8.1f}"
            else:
                row += f"{'  n/a':>8s}"
        print(row)

# Summary: which configs converged
print("\n" + "=" * 80)
print("VIABLE CONFIGURATIONS FOR CIM SWEEP (converged in exp0)")
print("=" * 80)

for mtype in sorted(baselines.keys()):
    any_results = any((mtype, wb, ib) in results for wb in WEIGHT_BITS for ib in INPUT_BITS)
    if not any_results:
        print(f"\nType {mtype}: {TYPE_NAMES[mtype]} — skipped (needs m=n)")
        continue

    converged = [(wb, ib, results[(mtype, wb, ib)]['iters'])
                 for wb in WEIGHT_BITS for ib in INPUT_BITS
                 if (mtype, wb, ib) in results and results[(mtype, wb, ib)]['converged']]

    print(f"\nType {mtype}: {TYPE_NAMES[mtype]} — {len(converged)} configs converged")
    if converged:
        # Show which (w, i) pairs work
        print(f"  {'w':>4s}  {'i':>4s}  {'iters':>6s}")
        for wb, ib, iters in sorted(converged):
            print(f"  {wb:4d}  {ib:4d}  {iters:6d}")

# Count for Table I configs specifically
print("\n" + "=" * 80)
print("TABLE I CONFIGS: {6, 8, 10} × {6, 8, 10}")
print("=" * 80)
table1_bits = [6, 8, 10]
for mtype in sorted(baselines.keys()):
    any_results = any((mtype, wb, ib) in results for wb in WEIGHT_BITS for ib in INPUT_BITS)
    if not any_results:
        print(f"  Type {mtype}: skipped")
        continue

    print(f"\n  Type {mtype}: {TYPE_NAMES[mtype]}")
    for wb in table1_bits:
        for ib in table1_bits:
            key = (mtype, wb, ib)
            if key in results:
                r = results[key]
                status = f"{r['iters']} iters" if r['converged'] else f"DIVERGED ||r||={r['final_residual']:.1e}"
                print(f"    w={wb:2d}, i={ib:2d}: {status}")
