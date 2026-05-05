"""
test_parallelism_noise.py — Compare the two ADC_output code paths.

Path A (output_noise > 0): digital matmul + flat Gaussian noise.
    read_noise, D2D, conductance are ALL BYPASSED.

Path B (output_noise == 0): full analog simulation.
    conductance lookup, D2D variation, read noise, V=IG, ADC sensing.

This test verifies the fix: set output_noise=0 to enable the analog path.

Run from NeuroSim root:
    python test_parallelism_noise.py
"""

import types
import torch
import numpy as np

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor


def make_cim_args(weight_precision=8, input_precision=8, parallel_read=128,
                  read_noise=0.0, output_noise=0.0, mem_states_file="",
                  model_name="pnoise_test"):
    return types.SimpleNamespace(
        input_precision=input_precision,
        weight_precision=weight_precision,
        adc_precision=7, dac_precision=1, bitcell=1,
        sub_array=[128, 128], parallel_read=parallel_read,
        mem_type="resistive",
        off_state=2.5e-5, on_state=3.33e-4,
        mem_states_file=mem_states_file,
        read_noise=read_noise, output_noise=output_noise,
        output_noise_file="",
        vdd=1.0, hardware=1,
        t=1, v=0.0, detect=0, target=0.0,
        rate_stuck_0=0.0, rate_stuck_1=0.0,
        model=model_name, batch_size=1, fake_quant=True,
        write_network=False, hook=False,
        quant_mode="adc", name="pnoise_layer",
        logger=None, calib=False,
    )


def create_cim_layer(W, cim_args):
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


def run_matmul(layer, v):
    inp = v.T.float().cuda()
    layer._input_quantizer.amax = inp.abs().max().item()
    with torch.no_grad():
        out = layer(inp)
    return out.T.double()


def run_trials(W, v, exact, cim_args, num_trials=50):
    """Run multiple matmuls and return error statistics."""
    layer = create_cim_layer(W, cim_args)
    errors = []
    for _ in range(num_trials):
        out = run_matmul(layer, v).squeeze()
        rel_err = (torch.linalg.norm(out - exact) / torch.linalg.norm(exact)).item()
        errors.append(rel_err)
    return np.mean(errors), np.std(errors)


# ── Tests ─────────────────────────────────────────────────────────

def test_code_paths():
    """Test 1: Verify the two code paths are different."""
    print("=" * 70)
    print("Test 1: Digital shortcut vs analog path (n=500, w=8, i=8, p=128)")
    print("=" * 70)

    n = 500
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = (W @ v).squeeze().cuda()

    configs = [
        ("No noise (either path)",     0.0,  0.0,  ""),
        ("Digital: out_noise=0.28",    0.0,  0.28, ""),
        ("Analog: read_noise=0.05",    0.05, 0.0,  "mem_states_rram.csv"),
        ("Analog: read+D2D",          0.05, 0.0,  "mem_states_rram.csv"),
    ]

    print(f"\n  {'Config':>30s}  {'mean_err':>12s}  {'std_err':>12s}  {'path':>8s}")
    print(f"  {'-'*70}")

    for label, rn, on, msf in configs:
        ca = make_cim_args(read_noise=rn, output_noise=on, mem_states_file=msf,
                           model_name=f"path_{label[:5]}")
        mean, std = run_trials(W, v, exact, ca)
        path = "digital" if on > 0 else "analog"
        print(f"  {label:>30s}  {mean:12.4e}  {std:12.4e}  {path:>8s}")

    print()


def test_analog_vs_parallelism():
    """Test 2: Does noise scale with p in the ANALOG path?"""
    print("=" * 70)
    print("Test 2: Analog path noise vs parallelism (n=500, w=8, i=8)")
    print("  read_noise=0.05, output_noise=0.0, D2D from mem_states_rram.csv")
    print("=" * 70)

    n = 500
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = (W @ v).squeeze().cuda()

    print(f"\n  {'p':>4s}  {'reads':>6s}  {'mean_err':>12s}  {'std_err':>12s}")
    print(f"  {'-'*45}")

    for pr in [16, 32, 64, 128]:
        ca = make_cim_args(parallel_read=pr, read_noise=0.05, output_noise=0.0,
                           mem_states_file="mem_states_rram.csv",
                           model_name=f"analog_p{pr}")
        mean, std = run_trials(W, v, exact, ca)
        n_reads = (n + pr - 1) // pr
        print(f"  {pr:4d}  {n_reads:6d}  {mean:12.4e}  {std:12.4e}")

    print()


def test_digital_vs_parallelism():
    """Test 3: Confirm digital path noise scaling (for comparison)."""
    print("=" * 70)
    print("Test 3: Digital path noise vs parallelism (n=500, w=8, i=8)")
    print("  read_noise=0.0, output_noise=0.28 (digital shortcut)")
    print("=" * 70)

    n = 500
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = (W @ v).squeeze().cuda()

    print(f"\n  {'p':>4s}  {'reads':>6s}  {'mean_err':>12s}  {'std_err':>12s}")
    print(f"  {'-'*45}")

    for pr in [16, 32, 64, 128]:
        ca = make_cim_args(parallel_read=pr, read_noise=0.0, output_noise=0.28,
                           mem_states_file="",
                           model_name=f"digital_p{pr}")
        mean, std = run_trials(W, v, exact, ca)
        n_reads = (n + pr - 1) // pr
        print(f"  {pr:4d}  {n_reads:6d}  {mean:12.4e}  {std:12.4e}")

    print()


def test_analog_noise_sources():
    """Test 4: Isolate read_noise vs D2D in analog path."""
    print("=" * 70)
    print("Test 4: Analog path — isolate noise sources (n=500, p=128)")
    print("=" * 70)

    n = 500
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)
    exact = (W @ v).squeeze().cuda()

    configs = [
        ("No noise",               0.0,  ""),
        ("D2D only",               0.0,  "mem_states_rram.csv"),
        ("Read noise only (0.05)", 0.05, ""),
        ("Read + D2D",             0.05, "mem_states_rram.csv"),
    ]

    for pr in [32, 128]:
        print(f"\n  p={pr} ({(n + pr - 1) // pr} reads):")
        print(f"  {'Config':>25s}  {'mean_err':>12s}  {'std_err':>12s}")
        print(f"  {'-'*55}")

        for label, rn, msf in configs:
            ca = make_cim_args(parallel_read=pr, read_noise=rn, output_noise=0.0,
                               mem_states_file=msf,
                               model_name=f"iso_{label[:5]}_p{pr}")
            mean, std = run_trials(W, v, exact, ca)
            print(f"  {label:>25s}  {mean:12.4e}  {std:12.4e}")

    print()


def test_analog_stochastic():
    """Test 5: Verify analog path noise is actually stochastic."""
    print("=" * 70)
    print("Test 5: Analog path stochasticity check (n=128, p=128)")
    print("=" * 70)

    n = 128
    torch.manual_seed(42)
    W = torch.randn(n, n, dtype=torch.float64)
    v = torch.randn(n, 1, dtype=torch.float64)

    ca = make_cim_args(parallel_read=128, read_noise=0.05, output_noise=0.0,
                       mem_states_file="mem_states_rram.csv",
                       model_name="stoch_test")
    layer = create_cim_layer(W, ca)

    results = []
    for t in range(5):
        out = run_matmul(layer, v).squeeze()
        results.append(out[:4].tolist())
        print(f"  Trial {t}: {out[:4].tolist()}")

    # Check if results are identical (deterministic D2D) or varying (stochastic read noise)
    all_same = all(r == results[0] for r in results)
    print(f"\n  All identical: {all_same}")
    if all_same:
        print("  WARNING: Analog noise appears deterministic — read_noise may not be active")
    else:
        print("  OK: Results vary across trials (stochastic noise confirmed)")

    print()


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}\n")

    test_code_paths()
    test_analog_vs_parallelism()
    test_digital_vs_parallelism()
    test_analog_noise_sources()
    test_analog_stochastic()

    print("=" * 70)
    print("Summary:")
    print("  - If Test 2 shows error INCREASING with p → analog path scales correctly")
    print("  - If Test 3 shows error DECREASING with p → confirms digital path bug")
    print("  - If Test 4 shows D2D and read noise both contribute → analog path works")
    print("  - If Test 5 shows variation across trials → stochastic noise confirmed")
    print()
    print("  FIX: set output_noise=0.0 in sweep_config.json to use analog path")
    print("  Then read_noise and D2D variation from mem_states actually take effect")
    print("=" * 70)
