"""
experiment_fp16_baseline.py — FP16 digital baseline using PETRA cost model.

Runs the same solver with FP16 arithmetic (torch.float16 cast, no custom
quantization) and computes total energy/latency using PETRA's published
silicon measurements.

The inner loop matmul casts both operands to FP16, computes in half
precision, and casts back to FP64 for the outer loop. On CUDA with
tensor cores, accumulation happens in FP32 — matching Haidar's FP16-TC
path and what modern FP16 hardware (including PETRA) does.

This gives a direct apples-to-apples comparison:
  - Same solver algorithm (unpreconditioned GMRES with diagonal preconditioner)
  - Same matrix types and sizes
  - Same convergence threshold
  - FP16 arithmetic (no analog noise, no ADC, no device variation)
  - PETRA cost model instead of NeuroSim PPA

PETRA reference:
  Cho et al., "PETRA: A 22nm 6.97TFLOPS/W AIB-Enabled Configurable Matrix
  and Convolution Accelerator," VLSI Circuits 2021.
  Node: 22nm Intel FinFET (same as NeuroSim configs)
  Array: 1024 FP16 MACs

Usage:
    python experiment_fp16_baseline.py
    python experiment_fp16_baseline.py --config sweep_config.json
"""

import os
import json
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator


# ── PETRA cost model ──────────────────────────────────────────────
# 500×500 MVM = 250,000 MACs
# Latency = 250,000 / (1024 MACs × frequency)
# Energy = 250,000 × energy_per_MAC

PETRA_CONFIGS = {
    "petra_nominal": {
        "label": "PETRA 0.88V (nominal)",
        "voltage": 0.88,
        "frequency_mhz": 701,
        "array_macs": 1024,
        "energy_per_mac_pJ": 0.89,
        "area_mm2": 3.04,
    },
    "petra_efficient": {
        "label": "PETRA 0.70V (peak efficiency)",
        "voltage": 0.70,
        "frequency_mhz": 416,
        "array_macs": 1024,
        "energy_per_mac_pJ": 0.29,
        "area_mm2": 3.04,
    },
}


def compute_petra_mvm_cost(n, petra_cfg):
    """Compute energy and latency for one n×n MVM."""
    n_macs = n * n  # 500×500 = 250,000
    energy_pJ = n_macs * petra_cfg["energy_per_mac_pJ"]
    latency_ns = n_macs / (petra_cfg["array_macs"] * petra_cfg["frequency_mhz"] * 1e-3)
    return energy_pJ, latency_ns


# ── FP16 matmul (cast to half, compute, cast back) ───────────────

def make_fp16_matmul():
    """FP16 matmul — cast both operands to torch.float16, compute, cast back.
    
    This matches what hardware FP16 units (PETRA, GPU tensor cores) do.
    On CUDA, PyTorch uses tensor core accumulation in FP32, which matches
    Haidar's FP16-TC path — the more relevant modern comparison.
    """
    def fn(A, x):
        return (A.half() @ x.half()).double()
    return fn


# ── Main ──────────────────────────────────────────────────────────

GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"

TYPE_NAMES = {
    1: "Diag dominant", 2: "Log-uniform σ, +λ",
    3: "Clustered σ, +λ", 5: "Arithmetic σ, +λ",
}


def run_baseline(cfg):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = cfg["matrix_size"]
    cond = cfg["cond_number"]
    max_outer = cfg["max_outer"]
    inner_iters = cfg["inner_iters"]
    tol = cfg["tol"]
    seed = cfg["seed"]
    sweep = cfg["sweep_configs"]

    gen = TestMatrixGenerator(n, seed=seed, device=GPU_DEVICE)
    torch.manual_seed(seed + 1)
    b = torch.randn(n, dtype=torch.float64, device=GPU_DEVICE)

    matmul_fn = make_fp16_matmul()

    # Precompute PETRA costs per MVM
    petra_costs = {}
    for name, pcfg in PETRA_CONFIGS.items():
        e, l = compute_petra_mvm_cost(n, pcfg)
        petra_costs[name] = {"energy_pJ": e, "latency_ns": l}
        print(f"  {pcfg['label']}: {e/1e3:.1f} nJ/MVM, {l/1e3:.3f} µs/MVM")
    print()

    results = {}

    for mtype in sorted(sweep.keys()):
        scfg = sweep[mtype]

        A = gen.type1() if mtype == 1 else \
            {2: gen.type2, 3: gen.type3, 5: gen.type5}[mtype](cond=cond)

        x_ref = torch.linalg.solve(A, b)

        # Run solver with FP16 quantized matmul
        solver = MixedPrecisionSolver(A, b, matmul_fn=matmul_fn, precond=scfg["precond"])
        x, hist = solver.solve(max_outer=max_outer, inner_iters=inner_iters,
                                tol=tol, verbose=False)

        rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
        converged = hist[-1] < tol * 10
        total_matmuls = len(hist) * inner_iters

        # Also run FP64 baseline (no quantization at all)
        solver_fp64 = MixedPrecisionSolver(A, b, precond=scfg["precond"])
        x64, hist64 = solver_fp64.solve(max_outer=max_outer, inner_iters=inner_iters,
                                         tol=tol, verbose=False)

        results[mtype] = {
            "history": hist,
            "iters": len(hist),
            "final_residual": hist[-1],
            "rel_error": rel_err,
            "converged": converged,
            "stop_reason": solver.stop_reason,
            "total_matmuls": total_matmuls,
            "fp64_history": hist64,
            "fp64_iters": len(hist64),
            "petra_costs": {},
        }

        # Compute total cost for each PETRA config
        for pname, pcost in petra_costs.items():
            results[mtype]["petra_costs"][pname] = {
                "energy_per_mvm_pJ": pcost["energy_pJ"],
                "latency_per_mvm_ns": pcost["latency_ns"],
                "total_energy_pJ": total_matmuls * pcost["energy_pJ"],
                "total_latency_ns": total_matmuls * pcost["latency_ns"],
                "total_energy_uJ": total_matmuls * pcost["energy_pJ"] / 1e6,
                "total_latency_us": total_matmuls * pcost["latency_ns"] / 1e3,
                "area_mm2": PETRA_CONFIGS[pname]["area_mm2"],
            }

        print(f"Type {mtype} ({TYPE_NAMES[mtype]}):")
        print(f"  FP16: {len(hist)} iters, {total_matmuls} matmuls, ||r||={hist[-1]:.2e}  "
              f"{'✓' if converged else '✗'}")
        print(f"  FP64: {len(hist64)} iters, ||r||={hist64[-1]:.2e}")
        for pname, pcfg in PETRA_CONFIGS.items():
            pc = results[mtype]["petra_costs"][pname]
            print(f"  {pcfg['label']}: E={pc['total_energy_uJ']:.3f} µJ, "
                  f"L={pc['total_latency_us']:.1f} µs")
        print()

    # Save
    save_path = os.path.join(OUTPUT_DIR, "fp16_baseline_results.pt")
    torch.save({
        "results": results,
        "petra_configs": PETRA_CONFIGS,
        "config": cfg,
    }, save_path)
    print(f"Results saved to {save_path}")

    print_summary(results)


def print_summary(results):
    print(f"\n{'='*90}")
    print("FP16 BASELINE SUMMARY (PETRA cost model)")
    print(f"{'='*90}")

    print(f"\n{'Type':>5s} {'Name':>22s} {'iters':>6s} {'matmuls':>8s} {'||r||':>10s} "
          f"{'E_nom(µJ)':>10s} {'L_nom(µs)':>10s} {'E_eff(µJ)':>10s} {'L_eff(µs)':>10s} {'conv':>5s}")
    print('-' * 100)

    for mtype in sorted(results.keys()):
        r = results[mtype]
        pc_nom = r["petra_costs"]["petra_nominal"]
        pc_eff = r["petra_costs"]["petra_efficient"]
        conv = '✓' if r['converged'] else '✗'

        print(f"{mtype:5d} {TYPE_NAMES[mtype]:>22s} {r['iters']:6d} {r['total_matmuls']:8d} "
              f"{r['final_residual']:10.2e} "
              f"{pc_nom['total_energy_uJ']:10.3f} {pc_nom['total_latency_us']:10.1f} "
              f"{pc_eff['total_energy_uJ']:10.3f} {pc_eff['total_latency_us']:10.1f} "
              f"{conv:>5s}")

    print(f"\n  PETRA nominal:  0.88V, 701 MHz, 0.89 pJ/MAC")
    print(f"  PETRA efficient: 0.70V, 416 MHz, 0.29 pJ/MAC")
    print(f"  Both: 1024 MACs, 22nm FinFET, 3.04 mm²")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sweep_config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cfg["sweep_configs"] = {int(k): v for k, v in cfg["sweep_configs"].items()}

    run_baseline(cfg)
