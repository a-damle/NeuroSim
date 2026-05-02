"""
test_neurosim_integration.py — End-to-end test of solver + NeuroSim CIM simulation.

Run from the NeuroSim root directory:
    cd NeuroSim-2DInferenceV1.5-dev
    python test_neurosim_integration.py

Prerequisites:
  - Fixed hook.py installed
  - write_layer imported in macro.py:
      from pytorch_quantization.utils.hook import write_layer
  - C++ backend compiled (cd NeuroSIM && make)
"""

import os
import sys
import glob
import types
import subprocess
import numpy as np
import torch

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.utils import hook

from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator
from acim_matmul import acim_matmul


# ── NeuroSim CIM wrapper ─────────────────────────────────────────

def make_cim_args(weight_precision=8, input_precision=8, parallel_read=128,
                  model="solver_test"):
    return types.SimpleNamespace(
        input_precision=input_precision,
        weight_precision=weight_precision,
        adc_precision=7, dac_precision=1, bitcell=1,
        sub_array=[128, 128], parallel_read=parallel_read,
        mem_type="resistive", off_state=6e-3, on_state=6e-3 * 17,
        mem_states_file="", read_noise=0.0, output_noise=0.0,
        output_noise_file="", vdd=1.0, hardware=1,
        t=1, v=0.0, detect=0, target=0.0,
        rate_stuck_0=0.0, rate_stuck_1=0.0,
        model=model, batch_size=1, fake_quant=True,
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


def make_cim_matmul_fn(layer):
    def fn(W_ignored, v):
        inp = v.T.float().cuda()
        layer._input_quantizer.amax = inp.abs().max().item()
        with torch.no_grad():
            out = layer(inp)
        return out.T.double().to(v.device)
    return fn


# ── Tests ─────────────────────────────────────────────────────────

def test_cim_matmul_correctness():
    print("=" * 60)
    print("Test 1: CIM matmul wrapper correctness")
    print("=" * 60)

    n = 16
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)

    exact = W @ v
    cim_args = make_cim_args()
    layer = create_cim_layer(W, cim_args)
    cim_result = make_cim_matmul_fn(layer)(W, v)

    rel_err = (torch.linalg.norm(cim_result - exact) / torch.linalg.norm(exact)).item()
    print(f"  Exact norm:     {torch.linalg.norm(exact).item():.4e}")
    print(f"  CIM norm:       {torch.linalg.norm(cim_result).item():.4e}")
    print(f"  Relative error: {rel_err:.4e}")
    ok = rel_err < 0.5
    print(f"  Result: {'PASS ✓' if ok else 'FAIL ✗'}\n")
    return ok


def test_solver_with_cim():
    print("=" * 60)
    print("Test 2: Solver convergence with CIM (Type 1, n=100)")
    print("=" * 60)

    device = torch.device('cuda')
    n = 100
    gen = TestMatrixGenerator(n, seed=42, device=device)
    A = gen.type1()
    torch.manual_seed(99)
    b = torch.randn(n, dtype=torch.float64, device=device)

    solver_tmp = MixedPrecisionSolver(A, b, precond='diag')
    W_stat = solver_tmp.W_stationary

    cim_args = make_cim_args(weight_precision=10, input_precision=10)
    layer = create_cim_layer(W_stat, cim_args)

    solver = MixedPrecisionSolver(A, b, matmul_fn=make_cim_matmul_fn(layer), precond='diag')
    x, hist = solver.solve(max_outer=50, inner_iters=30, tol=1e-10)

    x_ref = torch.linalg.solve(A, b)
    rel_err = (torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)).item()
    converged = hist[-1] < 1e-8

    print(f"\n  Final residual: {hist[-1]:.4e}")
    print(f"  Rel error:      {rel_err:.4e}")
    print(f"  Iterations:     {len(hist)}")
    print(f"  Result: {'PASS ✓' if converged else 'FAIL ✗'}\n")
    return converged


def test_ppa_extraction():
    print("=" * 60)
    print("Test 3: PPA trace generation and estimation")
    print("=" * 60)

    if not os.path.isfile('./NeuroSIM/main'):
        print("  SKIPPED — C++ backend not compiled")
        return None

    n = 128
    model_name = "ppa_solver_test"
    record_dir = f'./layer_record_{model_name}'
    net_csv = f'./NeuroSIM/NetWork_{model_name}.csv'

    # Clean up from previous runs
    import shutil
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.makedirs(record_dir)
    if os.path.exists(net_csv):
        os.remove(net_csv)

    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)

    # Create layer, then enable hook/write_network on the instance
    cim_args = make_cim_args(model=model_name)
    layer = create_cim_layer(W, cim_args)
    layer._cim_args.hook = True
    layer._cim_args.write_network = True

    # Write trace_command.sh header
    trace_cmd_path = os.path.join(record_dir, 'trace_command.sh')
    with open(trace_cmd_path, 'w') as f:
        f.write(f'./NeuroSIM/main ./NeuroSIM/NetWork_{model_name}.csv '
                f'{cim_args.weight_precision} {cim_args.input_precision} '
                f'{cim_args.sub_array[0]} {cim_args.parallel_read} ')

    # Run one matmul — triggers write_network + write_layer
    print("  Running CIM matmul to generate traces...")
    inp = v.T.float().cuda()
    layer._input_quantizer.amax = inp.abs().max().item()
    with torch.no_grad():
        out = layer(inp)
    print(f"  Output norm: {torch.linalg.norm(out).item():.4e}")

    # Find trace files by glob pattern (name may have leading underscore)
    weight_files = glob.glob(os.path.join(record_dir, 'weight*.csv'))
    input_files = glob.glob(os.path.join(record_dir, 'input*.csv'))

    if weight_files and input_files:
        w_shape = np.loadtxt(weight_files[0], delimiter=',').shape
        i_shape = np.loadtxt(input_files[0], delimiter=',').shape
        print(f"  Weight trace: {weight_files[0]} shape={w_shape}")
        print(f"  Input trace:  {input_files[0]} shape={i_shape}")
    else:
        print(f"  ERROR: Trace files not generated")
        print(f"  Directory contents: {os.listdir(record_dir)}")
        return False

    # Check NetWork CSV
    if os.path.exists(net_csv):
        with open(net_csv) as f:
            print(f"  Network CSV: {f.read().strip()}")
    else:
        print(f"  ERROR: {net_csv} not generated")
        return False

    # trace_command.sh should already have filenames appended by write_layer
    with open(trace_cmd_path) as f:
        cmd = f.read().strip()
    print(f"  Trace command: {cmd[:80]}...")

    # Run C++ PPA backend
    print(f"\n  Running C++ PPA backend...")
    result = subprocess.run(['/bin/bash', trace_cmd_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        print(f"  STDOUT: {result.stdout[:500]}")
        print(f"  Result: FAIL ✗\n")
        return False

    if any(k in result.stdout for k in ["Total", "Energy", "Latency", "Area"]):
        for line in result.stdout.split('\n'):
            if any(k in line for k in ["Total", "Energy", "Latency", "Area", "TOPS"]):
                print(f"    {line.strip()}")
        print(f"\n  Result: PASS ✓\n")
        return True
    else:
        print(f"  Output: {result.stdout[:500]}")
        print(f"  Result: FAIL ✗\n")
        return False


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}\n")

    t1 = test_cim_matmul_correctness()
    t2 = test_solver_with_cim()
    t3 = test_ppa_extraction()

    print("=" * 60)
    for i, t in [(1, t1), (2, t2), (3, t3)]:
        print(f"  Test {i}: {'PASS ✓' if t else ('FAIL ✗' if t == False else 'SKIP')}")

    if all(v for v in [t1, t2, t3] if v is not None):
        print("\nAll tests passed. Ready for Experiments 1-2.")
    else:
        print("\nSome tests failed. Fix and re-run.")
    print("=" * 60)
