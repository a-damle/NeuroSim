"""
experiment_1_2.py — CIM convergence and energy-performance tradeoff.

Reads sweep parameters, device noise, and K-averaging from sweep_config.json.

Usage:
    python experiment_1_2.py --device rram
    python experiment_1_2.py --device rram --config sweep_config_type1.json --suffix _type1
"""

import os
import sys
import glob
import json
import types
import shutil
import subprocess
import argparse
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor

from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator


GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"


# ── Config ────────────────────────────────────────────────────────

def load_config(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["sweep_configs"] = {int(k): v for k, v in cfg["sweep_configs"].items()}
    return cfg


# ── NeuroSim helpers ──────────────────────────────────────────────

def make_cim_args(weight_precision, input_precision, parallel_read,
                  model_name, dev_params):
    return types.SimpleNamespace(
        input_precision=input_precision,
        weight_precision=weight_precision,
        adc_precision=7, dac_precision=1, bitcell=1,
        sub_array=[128, 128], parallel_read=parallel_read,
        mem_type=dev_params["mem_type"],
        off_state=dev_params["off_state"],
        on_state=dev_params["on_state"],
        mem_states_file=dev_params["mem_states_file"],
        read_noise=dev_params["read_noise"],
        output_noise=dev_params["output_noise"],
        output_noise_file="",
        vdd=1.0, hardware=1,
        t=1, v=0.0, detect=0, target=0.0,
        rate_stuck_0=dev_params["rate_stuck_0"],
        rate_stuck_1=dev_params["rate_stuck_1"],
        model=model_name, batch_size=1, fake_quant=True,
        write_network=False, hook=False,
        quant_mode="adc", name="solver_layer",
        logger=None, calib=False,
    )


def create_cim_layer(W, cim_args):
    n = W.shape[0]
    cim.CIMLinear.set_default_quant_desc_input(
        QuantDescriptor(num_bits=cim_args.input_precision, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_weight(
        QuantDescriptor(num_bits=cim_args.weight_precision, axis=0, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_adc(
        QuantDescriptor(num_bits=cim_args.adc_precision, fake_quant=True))
    cim.CIMLinear.set_default_cim_args(cim_args)

    layer = cim.CIMLinear(n, n, bias=False).cuda()
    layer.weight.data = W.float().cuda()
    layer._weight_quantizer.enable()
    layer._input_quantizer.enable()
    layer._adc_quantizer.enable()
    layer._weight_quantizer.amax = W.float().abs().amax(dim=1, keepdim=True).cuda()
    return layer


def make_cim_matmul_fn(layer, K=1):
    """CIM matmul with K-device averaging."""
    def fn(W_ignored, v):
        inp = v.T.float().cuda()
        layer._input_quantizer.amax = inp.abs().max().item()

        with torch.no_grad():
            if K == 1:
                out = layer(inp)
            else:
                total = torch.zeros_like(inp).cuda()
                for _ in range(K):
                    total += layer(inp)
                out = total / K

        return out.T.double().to(v.device)
    return fn


def extract_ppa(layer, cim_args, v_sample):
    """Run one PPA call. Returns dict or None."""
    model_name = cim_args.model
    record_dir = f'./layer_record_{model_name}'
    net_csv = f'./NeuroSIM/NetWork_{model_name}.csv'

    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.makedirs(record_dir)
    if os.path.exists(net_csv):
        os.remove(net_csv)

    trace_cmd_path = os.path.join(record_dir, 'trace_command.sh')
    with open(trace_cmd_path, 'w') as f:
        f.write(f'./NeuroSIM/main ./NeuroSIM/NetWork_{model_name}.csv '
                f'{cim_args.weight_precision} {cim_args.input_precision} '
                f'{cim_args.sub_array[0]} {cim_args.parallel_read} ')

    layer._cim_args.hook = True
    layer._cim_args.write_network = True

    inp = v_sample.T.float().cuda()
    layer._input_quantizer.amax = inp.abs().max().item()
    with torch.no_grad():
        layer(inp)

    result = subprocess.run(['/bin/bash', trace_cmd_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None

    ppa = {}
    for line in result.stdout.split('\n'):
        if "readDynamicEnergy is:" in line and "layer1" in line:
            match = re.search(r'([\d.e+-]+)pJ', line)
            if match:
                ppa['energy_pJ'] = float(match.group(1))
        if "readLatency is:" in line and "layer1" in line:
            match = re.search(r'([\d.e+-]+)ns', line)
            if match:
                ppa['latency_ns'] = float(match.group(1))
        if "ChipArea" in line:
            match = re.search(r'([\d.e+-]+)um', line)
            if match:
                ppa['area_um2'] = float(match.group(1))
        if "Energy Efficiency TOPS/W" in line:
            match = re.search(r'([\d.e+-]+)', line.split(':')[-1])
            if match:
                ppa['tops_per_w'] = float(match.group(1))

    return ppa if 'energy_pJ' in ppa else None


# ── Main ──────────────────────────────────────────────────────────

def run_experiment(device_type, cfg, suffix=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = cfg["matrix_size"]
    cond = cfg["cond_number"]
    max_outer = cfg["max_outer"]
    inner_iters = cfg["inner_iters"]
    tol = cfg["tol"]
    seed = cfg["seed"]
    row_par = cfg["row_parallelism"]
    K_values = cfg.get("K_values", [1])
    sweep = cfg["sweep_configs"]
    dev_params = cfg["device_params"][device_type]

    gen = TestMatrixGenerator(n, seed=seed, device=GPU_DEVICE)
    torch.manual_seed(seed + 1)
    b = torch.randn(n, dtype=torch.float64, device=GPU_DEVICE)

    # Precompute
    matrices = {}
    w_stationary = {}
    x_refs = {}
    baselines = {}

    for mtype, scfg in sweep.items():
        A = gen.type1() if mtype == 1 else \
            {2: gen.type2, 3: gen.type3, 5: gen.type5}[mtype](cond=cond)
        matrices[mtype] = A
        x_refs[mtype] = torch.linalg.solve(A, b)
        solver_tmp = MixedPrecisionSolver(A, b, precond=scfg["precond"])
        w_stationary[mtype] = solver_tmp.W_stationary

        # FP64 baseline (no CIM noise)
        solver_bl = MixedPrecisionSolver(A, b, precond=scfg["precond"])
        x_bl, hist_bl = solver_bl.solve(max_outer=max_outer, inner_iters=inner_iters,
                                         tol=tol, verbose=False)
        baselines[mtype] = {"history": hist_bl, "iters": len(hist_bl),
                            "final_residual": hist_bl[-1],
                            "stop_reason": solver_bl.stop_reason}
        print(f"  FP64 baseline Type {mtype}: {len(hist_bl)} iters, ||r||={hist_bl[-1]:.2e}")

    total_runs = sum(
        len(s["w_bits"]) * len(s["i_bits"]) * len(row_par) * len(K_values)
        for s in sweep.values()
    )

    print(f"Device: {device_type.upper()}")
    print(f"Matrix size: {n}, cond: {cond:.0e}")
    print(f"Noise: read_noise={dev_params['read_noise']}, output_noise={dev_params['output_noise']}")
    print(f"K values: {K_values}")
    print(f"Total configurations: {total_runs}")
    print(f"Config: max_outer={max_outer}, inner_iters={inner_iters}, tol={tol:.0e}\n")

    results = {}
    count = 0

    for mtype, scfg in sweep.items():
        A = matrices[mtype]
        W_stat = w_stationary[mtype]
        x_ref = x_refs[mtype]

        print(f"--- Type {mtype}: {scfg['name']} ---")

        for wb in scfg["w_bits"]:
            for ib in scfg["i_bits"]:
                # PPA only depends on precision + parallelism, not K
                # Extract once per (wb, ib, pr), reuse for all K
                ppa_cache = {}

                for pr in row_par:
                    prev_residuals = {}  # track residuals across K for this (wb, ib, pr)

                    for K in K_values:
                        count += 1

                        # Skip higher K if lower K values show no improvement trend
                        if len(prev_residuals) >= 2:
                            sorted_ks = sorted(prev_residuals.keys())
                            r_low = prev_residuals[sorted_ks[-2]]
                            r_high = prev_residuals[sorted_ks[-1]]
                            # If doubling K didn't improve residual by at least 10×, skip
                            if r_high > r_low * 0.1 and r_high > tol * 10:
                                key = (mtype, wb, ib, pr, K)
                                results[key] = {
                                    "history": [], "iters": 0,
                                    "final_residual": float('inf'),
                                    "rel_error": float('inf'),
                                    "converged": False,
                                    "stop_reason": "skipped",
                                    "total_matmuls": 0, "K": K,
                                    "ppa_single": None,
                                    "total_energy_pJ": None,
                                    "total_latency_ns": None,
                                    "total_area_um2": None,
                                }
                                print(f"  [{count:3d}/{total_runs}] w={wb:2d} i={ib:2d} "
                                      f"p={pr:3d} K={K}: SKIPPED (K={sorted_ks[-1]} "
                                      f"showed insufficient improvement)")
                                prev_residuals[K] = float('inf')
                                continue

                        model_name = f"exp12_{device_type}_t{mtype}_w{wb}_i{ib}_p{pr}_k{K}"

                        ca = make_cim_args(wb, ib, pr, model_name, dev_params)
                        layer = create_cim_layer(W_stat, ca)

                        # Solver with K-averaged matmul
                        solver = MixedPrecisionSolver(
                            A, b, matmul_fn=make_cim_matmul_fn(layer, K=K),
                            precond=scfg["precond"])
                        x, hist = solver.solve(max_outer=max_outer,
                                                inner_iters=inner_iters,
                                                tol=tol, verbose=False)

                        rel_err = (torch.linalg.norm(x - x_ref) /
                                   torch.linalg.norm(x_ref)).item()
                        converged = hist[-1] < tol * 10
                        stop_reason = solver.stop_reason

                        # PPA: extract once per (wb, ib, pr), cache it
                        ppa_key = (wb, ib, pr)
                        if ppa_key not in ppa_cache:
                            v_sample = torch.randn(n, 1, dtype=torch.float64,
                                                    device=GPU_DEVICE)
                            ppa_cache[ppa_key] = extract_ppa(layer, ca, v_sample)

                        ppa = ppa_cache[ppa_key]

                        # Scale PPA by K (parallel read: K× energy, same latency, K× area)
                        total_matmuls = len(hist) * inner_iters
                        if ppa:
                            total_energy = total_matmuls * ppa['energy_pJ'] * K
                            total_latency = total_matmuls * ppa['latency_ns']
                            total_area = ppa['area_um2'] * K
                        else:
                            total_energy = None
                            total_latency = None
                            total_area = None

                        key = (mtype, wb, ib, pr, K)
                        results[key] = {
                            "history": hist,
                            "iters": len(hist),
                            "final_residual": hist[-1],
                            "rel_error": rel_err,
                            "converged": converged,
                            "stop_reason": stop_reason,
                            "total_matmuls": total_matmuls,
                            "K": K,
                            "ppa_single": ppa,
                            "total_energy_pJ": total_energy,
                            "total_latency_ns": total_latency,
                            "total_area_um2": total_area,
                        }

                        ppa_str = f"E={ppa['energy_pJ']*K:.0f}pJ/op" if ppa else "PPA FAIL"
                        status = "✓" if converged else stop_reason[:4]
                        print(f"  [{count:3d}/{total_runs}] w={wb:2d} i={ib:2d} "
                              f"p={pr:3d} K={K}: {len(hist):3d} iters  "
                              f"||r||={hist[-1]:.1e}  "
                              f"{status:>4s}  {ppa_str}")

                        prev_residuals[K] = hist[-1]

    # Save
    save_path = os.path.join(OUTPUT_DIR, f"exp12_{device_type}{suffix}_results.pt")
    torch.save({
        "results": results,
        "baselines": baselines,
        "config": cfg,
        "device_type": device_type,
        "device_params": dev_params,
    }, save_path)
    print(f"\nResults saved to {save_path}")

    print_summary(results, sweep, row_par, K_values, dev_params, device_type)
    plot_exp1_convergence(results, baselines, sweep, row_par, K_values, device_type, suffix)
    plot_exp2_pareto(results, baselines, sweep, row_par, K_values, device_type, suffix)


def print_summary(results, sweep, row_par, K_values, dev_params, device_type):
    print(f"\n{'='*90}")
    print(f"SUMMARY ({device_type.upper()})")
    print(f"Noise: read_noise={dev_params['read_noise']}, output_noise={dev_params['output_noise']}")
    print(f"{'='*90}")

    for mtype, scfg in sweep.items():
        print(f"\n  Type {mtype}: {scfg['name']}")
        print(f"  {'w':>3s} {'i':>3s} {'p':>4s} {'K':>3s}  {'iters':>5s}  {'||r||':>9s}  "
              f"{'E/op(pJ)':>9s}  {'TotalE(µJ)':>11s}  {'TotalL(µs)':>11s}  {'conv':>4s}")
        print(f"  {'-'*75}")

        for wb in scfg["w_bits"]:
            for ib in scfg["i_bits"]:
                for pr in row_par:
                    for K in K_values:
                        r = results[(mtype, wb, ib, pr, K)]
                        if r['ppa_single']:
                            e_op = f"{r['ppa_single']['energy_pJ']*K:.0f}"
                            t_e = f"{r['total_energy_pJ']/1e6:.3f}" if r['total_energy_pJ'] else "n/a"
                            t_l = f"{r['total_latency_ns']/1e3:.3f}" if r['total_latency_ns'] else "n/a"
                        else:
                            e_op = "n/a"
                            t_e = "n/a"
                            t_l = "n/a"
                        print(f"  {wb:3d} {ib:3d} {pr:4d} {K:3d}  {r['iters']:5d}  "
                              f"{r['final_residual']:9.1e}  {e_op:>9s}  {t_e:>11s}  {t_l:>11s}  "
                              f"{'✓' if r['converged'] else '✗':>4s}")


def plot_exp1_convergence(results, baselines, sweep, row_par, K_values, device_type, suffix=""):
    """Convergence curves. One subplot per type, lines colored by K, styled by precision."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    K_colors = {1: 'red', 4: 'blue', 8: 'green'}

    for idx, (mtype, scfg) in enumerate(sweep.items()):
        if idx >= 4:
            break
        ax = axes[idx]

        # FP64 baseline
        bl = baselines.get(mtype)
        if bl and bl["history"]:
            h = bl["history"]
            ax.semilogy(range(len(h)), h, 'k--', linewidth=2, label='FP64 baseline')

        # CIM results, p=max only
        pr = max(row_par)
        for K in K_values:
            for wb in scfg["w_bits"]:
                for ib in scfg["i_bits"]:
                    r = results.get((mtype, wb, ib, pr, K))
                    if r is None or r.get('stop_reason') == 'skipped':
                        continue
                    h = r["history"]
                    ax.semilogy(range(len(h)), h, color=K_colors.get(K, 'gray'),
                               linewidth=1.2, alpha=0.7,
                               label=f"w={wb},i={ib},K={K}")

        ax.set_title(f"Type {mtype}: {scfg['name']} (p={pr})", fontsize=10)
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("||r||₂")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, ncol=3, loc='upper right')

    plt.suptitle(f"Experiment 1: CIM Convergence ({device_type.upper()})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"exp1_{device_type}{suffix}_convergence.png")
    plt.savefig(path, dpi=150)
    print(f"Exp 1 plot saved to {path}")


def plot_exp2_pareto(results, baselines, sweep, row_par, K_values, device_type, suffix=""):
    """Pareto: total energy vs total latency, markers by K, colors by parallelism."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    K_markers = {1: 'o', 2: '^', 4: 's'}
    pr_colors = {32: 'red', 64: 'blue', 128: 'green'}

    for idx, (mtype, scfg) in enumerate(sweep.items()):
        if idx >= 4:
            break
        ax = axes[idx]

        for pr in row_par:
            for K in K_values:
                latencies = []
                energies = []
                labels = []

                for wb in scfg["w_bits"]:
                    for ib in scfg["i_bits"]:
                        r = results.get((mtype, wb, ib, pr, K))
                        if r is None or not r['converged']:
                            continue
                        if r['total_energy_pJ'] is None or r['total_latency_ns'] is None:
                            continue
                        latencies.append(r['total_latency_ns'] / 1e3)  # µs
                        energies.append(r['total_energy_pJ'] / 1e6)    # µJ
                        labels.append(f"w{wb}i{ib}")

                if latencies:
                    ax.scatter(latencies, energies,
                              marker=K_markers.get(K, 'o'),
                              color=pr_colors.get(pr, 'gray'),
                              s=60, alpha=0.8,
                              label=f"p={pr},K={K}")
                    for i, lbl in enumerate(labels):
                        ax.annotate(lbl, (latencies[i], energies[i]),
                                  fontsize=5, alpha=0.6,
                                  xytext=(3, 3), textcoords='offset points')

        ax.set_title(f"Type {mtype}: {scfg['name']}", fontsize=10)
        ax.set_xlabel("Total CIM latency (µs)")
        ax.set_ylabel("Total CIM energy (µJ)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, ncol=3)

    plt.suptitle(f"Experiment 2: Energy-Latency Pareto ({device_type.upper()})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"exp2_{device_type}{suffix}_pareto.png")
    plt.savefig(path, dpi=150)
    print(f"Exp 2 plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["sram", "rram"], default="rram")
    parser.add_argument("--config", default="sweep_config.json")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(args.device, cfg, suffix=args.suffix)
