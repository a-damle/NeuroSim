"""
test_k_tiling.py — Compare K-sequential averaging vs K-tiled averaging.

K-sequential: run K separate forward passes, average outputs.
K-tiled:      tile weight K times along output dim, one forward pass,
              reshape and average.

Tests:
  1. Both produce correct output shape
  2. Both reduce noise compared to K=1
  3. Noise statistics are comparable between approaches
  4. Solver convergence is comparable
  5. Speed comparison

Run from NeuroSim root:
    python test_k_tiling.py
"""

import time
import types
import torch
import numpy as np

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor

from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator


# ── CIM helpers ───────────────────────────────────────────────────

def make_cim_args(weight_precision=8, input_precision=8, parallel_read=128,
                  model_name="tile_test"):
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
        quant_mode="adc", name="tile_layer",
        logger=None, calib=False,
    )


def create_cim_layer(W, cim_args):
    """Create CIMLinear from weight matrix W (out_features, in_features)."""
    out_f, in_f = W.shape
    cim.CIMLinear.set_default_quant_desc_input(
        QuantDescriptor(num_bits=cim_args.input_precision, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_weight(
        QuantDescriptor(num_bits=cim_args.weight_precision, axis=0, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_adc(
        QuantDescriptor(num_bits=cim_args.adc_precision, fake_quant=True))
    cim.CIMLinear.set_default_cim_args(cim_args)

    layer = cim.CIMLinear(in_f, out_f, bias=False).cuda()
    layer.weight.data = W.float().cuda()
    layer._weight_quantizer.enable()
    layer._input_quantizer.enable()
    layer._adc_quantizer.enable()
    layer._weight_quantizer.amax = W.float().abs().amax(dim=1, keepdim=True).cuda()
    return layer


# ── K-sequential matmul (current approach) ────────────────────────

def make_sequential_matmul_fn(layer, K):
    """K separate forward passes, averaged."""
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


# ── K-tiled matmul (new approach) ─────────────────────────────────

def create_tiled_layer(W, K, cim_args):
    """Create a CIMLinear with W tiled K times along output dimension.

    Original W: shape (n, n)
    Tiled:      shape (K*n, n) — K independent copies stacked vertically.
    CIMLinear(n, K*n) computes input @ W_tiled.T → (1, K*n)
    """
    W_tiled = W.repeat(K, 1)  # (K*n, n)
    return create_cim_layer(W_tiled, cim_args)


def make_tiled_matmul_fn(layer_tiled, K, n):
    """One forward pass through tiled layer, reshape and average."""
    def fn(W_ignored, v):
        inp = v.T.float().cuda()  # (1, n)
        layer_tiled._input_quantizer.amax = inp.abs().max().item()

        with torch.no_grad():
            out_tiled = layer_tiled(inp)  # (1, K*n)

        # Reshape to (K, n) and average
        out_k = out_tiled.view(K, n)      # (K, n)
        out_avg = out_k.mean(dim=0, keepdim=True)  # (1, n)

        return out_avg.T.double().to(v.device)  # (n, 1)
    return fn


# ── Tests ─────────────────────────────────────────────────────────

def test_shape_and_basic():
    """Test 1: Both approaches produce correct output shape."""
    print("=" * 60)
    print("Test 1: Output shape and basic correctness")
    print("=" * 60)

    n = 128
    K = 4
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = W @ v

    cim_args = make_cim_args(model_name="tile_shape")

    # Sequential
    layer_seq = create_cim_layer(W, cim_args)
    fn_seq = make_sequential_matmul_fn(layer_seq, K)
    out_seq = fn_seq(W, v)

    # Tiled
    layer_tiled = create_tiled_layer(W, K, cim_args)
    fn_tiled = make_tiled_matmul_fn(layer_tiled, K, n)
    out_tiled = fn_tiled(W, v)

    print(f"  Exact shape:      {tuple(exact.shape)}")
    print(f"  Sequential shape: {tuple(out_seq.shape)}")
    print(f"  Tiled shape:      {tuple(out_tiled.shape)}")

    shape_ok = out_seq.shape == exact.shape == out_tiled.shape
    print(f"  Shapes match: {'PASS ✓' if shape_ok else 'FAIL ✗'}")

    err_seq = (torch.linalg.norm(out_seq - exact) / torch.linalg.norm(exact)).item()
    err_tiled = (torch.linalg.norm(out_tiled - exact) / torch.linalg.norm(exact)).item()
    print(f"  Sequential rel_err: {err_seq:.4e}")
    print(f"  Tiled rel_err:      {err_tiled:.4e}")
    print()

    return shape_ok


def test_noise_statistics():
    """Test 2: Both approaches have comparable noise statistics."""
    print("=" * 60)
    print("Test 2: Noise statistics comparison (20 trials)")
    print("=" * 60)

    n = 128
    num_trials = 20
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = W @ v

    cim_args = make_cim_args(model_name="tile_noise")

    for K in [1, 4, 8]:
        # Sequential
        layer_seq = create_cim_layer(W, cim_args)
        fn_seq = make_sequential_matmul_fn(layer_seq, K)
        errs_seq = []
        for _ in range(num_trials):
            out = fn_seq(W, v)
            errs_seq.append((torch.linalg.norm(out - exact) / torch.linalg.norm(exact)).item())

        # Tiled
        layer_tiled = create_tiled_layer(W, K, cim_args)
        fn_tiled = make_tiled_matmul_fn(layer_tiled, K, n)
        errs_tiled = []
        for _ in range(num_trials):
            out = fn_tiled(W, v)
            errs_tiled.append((torch.linalg.norm(out - exact) / torch.linalg.norm(exact)).item())

        print(f"  K={K}:")
        print(f"    Sequential: mean={np.mean(errs_seq):.4e} ± {np.std(errs_seq):.4e}")
        print(f"    Tiled:      mean={np.mean(errs_tiled):.4e} ± {np.std(errs_tiled):.4e}")

    print()


def test_solver_convergence():
    """Test 3: Solver convergence is comparable between approaches."""
    print("=" * 60)
    print("Test 3: Solver convergence comparison (Type 1, n=128)")
    print("=" * 60)

    device = torch.device('cuda')
    n = 128

    gen = TestMatrixGenerator(n, seed=42, device=device)
    A = gen.type1()
    torch.manual_seed(99)
    b = torch.randn(n, dtype=torch.float64, device=device)

    solver_tmp = MixedPrecisionSolver(A, b, precond='diag')
    W_stat = solver_tmp.W_stationary
    x_ref = torch.linalg.solve(A, b)

    cim_args = make_cim_args(weight_precision=8, input_precision=8,
                              model_name="tile_solver")

    for K in [1, 4, 8]:
        # Sequential
        layer_seq = create_cim_layer(W_stat, cim_args)
        fn_seq = make_sequential_matmul_fn(layer_seq, K)
        solver = MixedPrecisionSolver(A, b, matmul_fn=fn_seq, precond='diag')
        x_seq, hist_seq = solver.solve(max_outer=30, inner_iters=30, tol=1e-10, verbose=False)

        # Tiled
        layer_tiled = create_tiled_layer(W_stat, K, cim_args)
        fn_tiled = make_tiled_matmul_fn(layer_tiled, K, n)
        solver = MixedPrecisionSolver(A, b, matmul_fn=fn_tiled, precond='diag')
        x_tiled, hist_tiled = solver.solve(max_outer=30, inner_iters=30, tol=1e-10, verbose=False)

        print(f"  K={K}:")
        print(f"    Sequential: {len(hist_seq):3d} iters  ||r||={hist_seq[-1]:.2e}")
        print(f"    Tiled:      {len(hist_tiled):3d} iters  ||r||={hist_tiled[-1]:.2e}")

    print()


def test_speed():
    """Test 4: Speed comparison."""
    print("=" * 60)
    print("Test 4: Speed comparison (n=500, 100 matmuls)")
    print("=" * 60)

    n = 500
    num_ops = 100
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)

    cim_args = make_cim_args(model_name="tile_speed")

    for K in [1, 4, 8]:
        # Sequential
        layer_seq = create_cim_layer(W, cim_args)
        fn_seq = make_sequential_matmul_fn(layer_seq, K)
        fn_seq(W, v)  # warmup
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_ops):
            fn_seq(W, v)
        torch.cuda.synchronize()
        t_seq = time.perf_counter() - start

        # Tiled
        layer_tiled = create_tiled_layer(W, K, cim_args)
        fn_tiled = make_tiled_matmul_fn(layer_tiled, K, n)
        fn_tiled(W, v)  # warmup
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_ops):
            fn_tiled(W, v)
        torch.cuda.synchronize()
        t_tiled = time.perf_counter() - start

        speedup = t_seq / t_tiled if t_tiled > 0 else float('inf')
        print(f"  K={K}: sequential={t_seq:.3f}s  tiled={t_tiled:.3f}s  speedup={speedup:.2f}×")

    print()


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}\n")

    test_shape_and_basic()
    test_noise_statistics()
    test_solver_convergence()
    test_speed()

    print("=" * 60)
    print("If noise stats and convergence are comparable, and tiled is")
    print("faster, replace sequential with tiled in experiment_1_2.py")
    print("=" * 60)
