"""
plot_pareto.py — Energy vs Latency Pareto frontier.

Uses Le Gallo's convergence definition: converged when ||r|| < tol.
For each config, finds the first iteration crossing the threshold,
then computes total energy and latency up to that point.

Usage:
    python plot_pareto.py results/exp12_rram_results.pt
    python plot_pareto.py results/exp12_rram_results.pt --tol 1e-3
    python plot_pareto.py results/exp12_rram_results.pt --tol 1e-5
"""

import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def find_convergence_iter(history, tol):
    """Find the first iteration where residual drops below tol.
    Returns the iteration index (0-based), or None if never converges.
    """
    for i, res in enumerate(history):
        if res < tol:
            return i
    return None


def load_and_process(pt_path, tol):
    d = torch.load(pt_path, weights_only=False, map_location='cpu')
    results = d['results']
    baselines = d.get('baselines', {})
    cfg = d['config']
    inner_iters = cfg['inner_iters']
    device_type = d['device_type']

    processed = {}
    for key, r in results.items():
        mtype, wb, ib, pr, K = key

        conv_iter = find_convergence_iter(r['history'], tol)
        if conv_iter is None:
            continue  # didn't reach threshold

        ppa = r.get('ppa_single')
        if ppa is None:
            continue

        n_matmuls = (conv_iter + 1) * inner_iters  # +1 because iter 0 counts
        total_energy_uJ = n_matmuls * ppa['energy_pJ'] * K / 1e6
        total_latency_us = n_matmuls * ppa['latency_ns'] / 1e3
        total_area_mm2 = ppa['area_um2'] * K / 1e6

        processed[key] = {
            'conv_iter': conv_iter,
            'n_matmuls': n_matmuls,
            'total_energy_uJ': total_energy_uJ,
            'total_latency_us': total_latency_us,
            'total_area_mm2': total_area_mm2,
            'energy_per_op_pJ': ppa['energy_pJ'] * K,
            'latency_per_op_ns': ppa['latency_ns'],
            'final_residual': r['history'][conv_iter],
            'K': K,
            'w': wb, 'i': ib, 'p': pr,
        }

    # Process baselines
    bl_processed = {}
    for mtype, bl in baselines.items():
        conv_iter = find_convergence_iter(bl['history'], tol)
        if conv_iter is not None:
            bl_processed[mtype] = {
                'conv_iter': conv_iter,
                'n_matmuls': (conv_iter + 1) * inner_iters,
            }

    return processed, bl_processed, cfg, device_type


def plot_pareto(processed, bl_processed, cfg, device_type, tol, output_path):
    TYPE_NAMES = {
        1: "Diag dominant", 2: "Log-uniform σ, +λ",
        3: "Clustered σ, +λ", 5: "Arithmetic σ, +λ",
    }

    K_markers = {1: 'o', 4: '^', 8: 's'}
    K_sizes = {1: 40, 4: 55, 8: 70}
    pr_colors = {32: '#e41a1c', 64: '#377eb8', 128: '#4daf4a'}

    types_with_data = sorted(set(k[0] for k in processed.keys()))
    n_types = len(types_with_data)

    if n_types == 0:
        print(f"No configs converged at tol={tol:.0e}. Try a larger tolerance.")
        return

    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5), squeeze=False)
    axes = axes.flatten()

    for idx, mtype in enumerate(types_with_data):
        ax = axes[idx]

        entries = {k: v for k, v in processed.items() if k[0] == mtype}
        if not entries:
            continue

        # Plot each (pr, K) group
        for pr in sorted(set(v['p'] for v in entries.values())):
            for K in sorted(set(v['K'] for v in entries.values())):
                pts = [(v['total_latency_us'], v['total_energy_uJ'], v['w'], v['i'])
                       for k, v in entries.items()
                       if v['p'] == pr and v['K'] == K]

                if not pts:
                    continue

                lats, ens, ws, iis = zip(*pts)
                ax.scatter(lats, ens,
                          marker=K_markers.get(K, 'o'),
                          s=K_sizes.get(K, 40),
                          color=pr_colors.get(pr, 'gray'),
                          alpha=0.8,
                          label=f'p={pr}, K={K}',
                          edgecolors='black', linewidths=0.3)

                # Annotate each point
                for lat, en, w, i in pts:
                    ax.annotate(f'w{w}i{i}', (lat, en),
                              fontsize=5, alpha=0.6,
                              xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('Total CIM latency (µs)', fontsize=10)
        ax.set_ylabel('Total CIM energy (µJ)', fontsize=10)
        ax.set_title(f'Type {mtype}: {TYPE_NAMES.get(mtype, "?")}', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc='upper left')

    plt.suptitle(f'Energy vs Latency Pareto — {device_type.upper()}\n'
                 f'Convergence: ||r|| < {tol:.0e}  '
                 f'(n={cfg["matrix_size"]}, m={cfg["inner_iters"]})',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Plot saved to {output_path}')


def print_summary(processed, bl_processed, tol):
    """Print summary table of converged configs."""
    types = sorted(set(k[0] for k in processed.keys()))

    print(f'\nConverged configs at tol={tol:.0e}:')
    print(f'{"Type":>5s} {"w":>3s} {"i":>3s} {"p":>4s} {"K":>3s} '
          f'{"iter":>5s} {"matmuls":>8s} '
          f'{"E/op(pJ)":>9s} {"TotE(µJ)":>9s} {"TotL(µs)":>9s} '
          f'{"Area(mm²)":>9s} {"||r||":>9s}')
    print('-' * 85)

    for mtype in types:
        entries = sorted([(k, v) for k, v in processed.items() if k[0] == mtype],
                        key=lambda x: x[1]['total_energy_uJ'])
        for key, v in entries:
            _, wb, ib, pr, K = key
            print(f'{mtype:5d} {wb:3d} {ib:3d} {pr:4d} {K:3d} '
                  f'{v["conv_iter"]:5d} {v["n_matmuls"]:8d} '
                  f'{v["energy_per_op_pJ"]:9.1f} {v["total_energy_uJ"]:9.3f} '
                  f'{v["total_latency_us"]:9.1f} '
                  f'{v["total_area_mm2"]:9.3f} {v["final_residual"]:9.1e}')

        # Baseline comparison
        bl = bl_processed.get(mtype)
        if bl:
            print(f'  {"FP64":>11s} {"baseline":>6s}       '
                  f'{bl["conv_iter"]:5d} {bl["n_matmuls"]:8d}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pt_file', help='Path to exp12_*_results.pt')
    parser.add_argument('--tol', type=float, default=1e-5,
                        help='Convergence tolerance (default: 1e-5, Le Gallo used 1e-5 and 1e-3)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output plot path (default: auto-generated)')
    args = parser.parse_args()

    processed, bl_processed, cfg, device_type = load_and_process(args.pt_file, args.tol)

    if args.output is None:
        tol_str = f'{args.tol:.0e}'.replace('-', 'm')
        args.output = args.pt_file.replace('_results.pt', f'_pareto_tol{tol_str}.png')

    print(f'Loaded {args.pt_file}')
    print(f'Tolerance: {args.tol:.0e}')
    print(f'Configs that reached threshold: {len(processed)}')

    print_summary(processed, bl_processed, args.tol)
    plot_pareto(processed, bl_processed, cfg, device_type, args.tol, args.output)
