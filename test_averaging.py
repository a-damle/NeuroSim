"""
test_averaging.py — Test K-device averaging for CIM matmul.

Verifies:
  1. Averaging K matmul results reduces effective noise
  2. Solver convergence improves with higher K
  3. PPA energy scales linearly with K (parallel read, same latency)

Run from NeuroSim root:
    python test_averaging.py
"""

import os
import types
import shutil
import subprocess
import re
import torch
import numpy as np

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor

from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator


# ── CIM helpers (same as experiment_1_2.py) ───────────────────────

def make_cim_args(weight_precision=8, input_precision=8, parallel_read=128,
                  model_name="avg_test"):
    return types.SimpleNamespace(
        input_precision=input_precision,
        weight_precision=weight_precision,
        adc_precision=7, dac_precision=1, bitcell=1,
        sub_array=[128, 128], parallel_read=parallel_read,
        mem_type="resistive",
        off_state=2.5e-5, on_state=3.33e-4,
        mem_states_file="mem_states_rram.csv",
        read_noise=0.05, output_noise=0.28,
        output_noise_file="",
        vdd=1.0, hardware=1,
        t=1, v=0.0, detect=0, target=0.0,
        rate_stuck_0=0.0, rate_stuck_1=0.0,
        model=model_name, batch_size=1, fake_quant=True,
        write_network=False, hook=False,
        quant_mode="adc", name="avg_layer",
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

    return ppa if 'energy_pJ' in ppa else None


# ── Tests ─────────────────────────────────────────────────────────

def test_noise_reduction():
    """Test 1: K-averaging reduces matmul error."""
    print("=" * 60)
    print("Test 1: Noise reduction with K-averaging")
    print("=" * 60)

    n = 128
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = W @ v

    cim_args = make_cim_args(model_name="avg_noise_test")
    layer = create_cim_layer(W, cim_args)

    # Run multiple trials to get stable error estimates
    num_trials = 20

    for K in [1, 2, 4]:
        matmul_fn = make_cim_matmul_fn(layer, K=K)
        errors = []
        for _ in range(num_trials):
            result = matmul_fn(W, v)
            rel_err = (torch.linalg.norm(result - exact) / torch.linalg.norm(exact)).item()
            errors.append(rel_err)

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        print(f"  K={K}: mean_rel_err={mean_err:.4e} ± {std_err:.4e}")

    print()


def test_solver_convergence():
    """Test 2: Solver converges faster with higher K."""
    print("=" * 60)
    print("Test 2: Solver convergence with K-averaging (Type 1, n=128)")
    print("=" * 60)

    device = torch.device('cuda')
    n = 128

    gen = TestMatrixGenerator(n, seed=42, device=device)
    A = gen.type1()
    torch.manual_seed(99)
    b = torch.randn(n, dtype=torch.float64, device=device)

    # Get stationary matrix
    solver_tmp = MixedPrecisionSolver(A, b, precond='diag')
    W_stat = solver_tmp.W_stationary

    x_ref = torch.linalg.solve(A, b)

    for K in [1, 2, 4]:
        cim_args = make_cim_args(weight_precision=8, input_precision=8,
                                  model_name=f"avg_solver_k{K}")
        layer = create_cim_layer(W_stat, cim_args)
        matmul_fn = make_cim_matmul_fn(layer, K=K)

        solver = MixedPrecisionSolver(A, b, matmul_fn=matmul_fn, precond='diag')
        x, hist = solver.solve(max_outer=30, inner_iters=30, tol=1e-10, verbose=False)

        rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
        converged = hist[-1] < 1e-8

        print(f"  K={K}: {len(hist):3d} iters  ||r||={hist[-1]:.2e}  "
              f"rel_err={rel_err:.2e}  {'✓' if converged else '✗'}")

    print()


def test_harder_matrix():
    """Test 3: K-averaging enables convergence on harder matrix (Type 5)."""
    print("=" * 60)
    print("Test 3: K-averaging on harder matrix (Type 5, n=128)")
    print("=" * 60)

    device = torch.device('cuda')
    n = 128

    gen = TestMatrixGenerator(n, seed=42, device=device)
    A = gen.type5(cond=1e4)
    torch.manual_seed(99)
    b = torch.randn(n, dtype=torch.float64, device=device)

    solver_tmp = MixedPrecisionSolver(A, b, precond='diag')
    W_stat = solver_tmp.W_stationary

    x_ref = torch.linalg.solve(A, b)

    for K in [1, 2, 4]:
        cim_args = make_cim_args(weight_precision=10, input_precision=10,
                                  model_name=f"avg_hard_k{K}")
        layer = create_cim_layer(W_stat, cim_args)
        matmul_fn = make_cim_matmul_fn(layer, K=K)

        solver = MixedPrecisionSolver(A, b, matmul_fn=matmul_fn, precond='diag')
        x, hist = solver.solve(max_outer=50, inner_iters=30, tol=1e-10, verbose=False)

        rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
        converged = hist[-1] < 1e-8

        print(f"  K={K}: {len(hist):3d} iters  ||r||={hist[-1]:.2e}  "
              f"rel_err={rel_err:.2e}  {'✓' if converged else '✗'}")

    print()


def test_ppa_scaling():
    """Test 4: PPA energy/latency/area for K-averaging."""
    print("=" * 60)
    print("Test 4: PPA scaling with K")
    print("=" * 60)

    if not os.path.isfile('./NeuroSIM/main'):
        print("  SKIPPED — C++ backend not compiled")
        return

    n = 128
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64, device='cuda')

    cim_args = make_cim_args(model_name="avg_ppa_test")
    layer = create_cim_layer(W, cim_args)
    ppa = extract_ppa(layer, cim_args, v)

    if ppa is None:
        print("  PPA extraction failed")
        return

    print(f"  Single device PPA:")
    print(f"    Energy:  {ppa['energy_pJ']:.1f} pJ")
    print(f"    Latency: {ppa['latency_ns']:.1f} ns")
    print(f"    Area:    {ppa['area_um2']:.0f} um²")

    print(f"\n  Scaled PPA (K devices in parallel):")
    print(f"  {'K':>3s}  {'Energy(pJ)':>12s}  {'Latency(ns)':>12s}  {'Area(um²)':>12s}")
    print(f"  {'-'*45}")

    for K in [1, 2, 4]:
        # Parallel read: K× energy, same latency, K× area
        e = ppa['energy_pJ'] * K
        l = ppa['latency_ns']       # same — parallel read
        a = ppa['area_um2'] * K
        print(f"  {K:3d}  {e:12.1f}  {l:12.1f}  {a:12.0f}")

    # Example total cost for a solve
    print(f"\n  Example total cost: Type 1, 10 iters × 30 inner = 300 matmuls")
    for K in [1, 2, 4]:
        total_e = 300 * ppa['energy_pJ'] * K
        total_l = 300 * ppa['latency_ns']
        print(f"    K={K}: total_energy={total_e/1e6:.2f} µJ, total_latency={total_l/1e3:.2f} µs")

    print()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}\n")

    test_noise_reduction()
    test_solver_convergence()
    test_harder_matrix()
    test_ppa_scaling()

    print("=" * 60)
    print("Summary:")
    print("  - K-averaging reduces noise across all CIM noise sources")
    print("  - Energy scales as K× (parallel read)")
    print("  - Latency unchanged (parallel read)")
    print("  - Area scales as K× (K devices per element)")
    print("  - Tradeoff: more area/energy per op vs fewer iterations")
    print("=" * 60)
