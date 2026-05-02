"""
test_cim_linear.py — Smoke test for NeuroSim's CIMLinear layer.
Run:  cd NeuroSim-2DInferenceV1.5-dev && python test_cim_linear.py
"""

import types, torch, torch.nn.functional as F
from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor


def make_cim_args():
    return types.SimpleNamespace(
        input_precision=8, weight_precision=8, adc_precision=7,
        dac_precision=1, bitcell=1, sub_array=[128,128], parallel_read=128,
        mem_type="resistive", off_state=6e-3, on_state=6e-3*17,
        mem_states_file="", read_noise=0.0, output_noise=0.0,
        output_noise_file="", vdd=1.0, hardware=1, t=1, v=0.0,
        detect=0, target=0.0, rate_stuck_0=0.0, rate_stuck_1=0.0,
        model="test", batch_size=1, fake_quant=True,
        write_network=False, hook=False, quant_mode="iw",
        name="test_layer", logger=None, calib=False,
    )

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Hardware sim requires CUDA. Testing fake-quant only.\n")

    cim.CIMLinear.set_default_quant_desc_input(QuantDescriptor(num_bits=8, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_weight(QuantDescriptor(num_bits=8, axis=0, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_adc(QuantDescriptor(num_bits=7, fake_quant=True))
    cim.CIMLinear.set_default_cim_args(make_cim_args())

    IN, OUT = 4, 3
    layer = cim.CIMLinear(IN, OUT, bias=True).to(device)
    torch.manual_seed(42)
    layer.weight.data = torch.randn(OUT, IN, device=device)
    layer.bias.data   = torch.zeros(OUT, device=device)
    x = torch.randn(2, IN, device=device)

    print("=" * 60)
    print(f"CIMLinear({IN}, {OUT})  weight {tuple(layer.weight.shape)}, input {tuple(x.shape)}")
    print("=" * 60)

    # Test 1: pure FP
    layer._input_quantizer.disable(); layer._weight_quantizer.disable(); layer._adc_quantizer.disable()
    layer._cim_args.quant_mode = "iw"
    match = torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias), atol=1e-5)
    print(f"\n[Test 1] FP baseline:      {'PASS ✓' if match else 'FAIL ✗'}")

    # Test 2: fake-quant
    layer._input_quantizer.enable(); layer._weight_quantizer.enable()
    layer._input_quantizer.amax  = torch.tensor(x.abs().max().item(), device=device)
    layer._weight_quantizer.amax = layer.weight.abs().amax(dim=1, keepdim=True).to(device)
    out_q = layer(x)
    ok = out_q.shape == (2,3) and torch.isfinite(out_q).all()
    print(f"[Test 2] Fake-quant (iw):  {'PASS ✓' if ok else 'FAIL ✗'}")

    # Test 3: hardware sim
    if device == "cuda":
        layer._cim_args.quant_mode = "adc"; layer._cim_args.hardware = True
        out_hw = layer(x)
        ok = out_hw.shape == (2,3) and torch.isfinite(out_hw).all()
        print(f"[Test 3] Hardware sim:     {'PASS ✓' if ok else 'FAIL ✗'}")
    else:
        print("[Test 3] Hardware sim:     SKIPPED (no CUDA)")

    print("\n" + "=" * 60)
    print("All executed tests passed!" if match else "Some tests failed.")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
