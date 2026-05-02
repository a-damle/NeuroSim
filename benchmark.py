"""
benchmark.py — Compare CPU vs GPU performance for the solver.

Tests FP64 GMRES inner loop at different matrix sizes to determine
whether GPU acceleration helps (consumer GPUs have limited FP64).

Run:  python benchmark.py
"""

import time
import torch
from solver import MixedPrecisionSolver
from test_matrices import TestMatrixGenerator


def benchmark_one(n, device, inner_iters=30, max_outer=10):
    """Run solver on type 1 matrix and return wall-clock time."""
    gen = TestMatrixGenerator(n, seed=42, device=device)
    A = gen.type1()
    torch.manual_seed(99)
    b = torch.randn(n, dtype=torch.float64, device=device)

    # Warmup (especially important for GPU)
    solver = MixedPrecisionSolver(A, b, precond='diag')
    solver.solve(max_outer=2, inner_iters=inner_iters, tol=1e-15, verbose=False)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    solver = MixedPrecisionSolver(A, b, precond='diag')
    x, hist = solver.solve(max_outer=max_outer, inner_iters=inner_iters,
                            tol=1e-15, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed, len(hist)


def main():
    cpu = torch.device('cpu')
    has_gpu = torch.cuda.is_available()
    gpu = torch.device('cuda') if has_gpu else None

    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Check FP64 capability
        props = torch.cuda.get_device_properties(0)
        print(f"     Compute capability: {props.major}.{props.minor}")
        print(f"     SM count: {props.multi_processor_count}")
        print()

    sizes = [100, 200, 500]
    inner_iters = 30
    max_outer = 20

    print(f"{'n':>6s}  {'CPU (s)':>10s}  {'iters':>6s}", end="")
    if has_gpu:
        print(f"  {'GPU (s)':>10s}  {'iters':>6s}  {'speedup':>8s}", end="")
    print()
    print("-" * (40 + (30 if has_gpu else 0)))

    for n in sizes:
        t_cpu, it_cpu = benchmark_one(n, cpu, inner_iters, max_outer)
        row = f"{n:6d}  {t_cpu:10.3f}  {it_cpu:6d}"

        if has_gpu:
            t_gpu, it_gpu = benchmark_one(n, gpu, inner_iters, max_outer)
            speedup = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            row += f"  {t_gpu:10.3f}  {it_gpu:6d}  {speedup:7.2f}x"

        print(row)

    if has_gpu:
        print(f"\nNote: FP64 throughput on consumer GPUs (RTX) is 1/32 of FP32.")
        print(f"      If speedup < 1, use CPU for experiment 0.")


if __name__ == "__main__":
    main()
