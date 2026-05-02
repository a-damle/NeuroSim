"""
test_ppa.py — End-to-end PPA estimation test for NeuroSim v1.5.

Prerequisites:
  1. Replace hook.py:  cp hook.py pytorch-quantization/pytorch_quantization/utils/hook.py
  2. Compile C++:      cd NeuroSIM && make && cd ..

Run:  python test_ppa.py

C++ trace format (v1.5):
  Weight CSV:  float values in [-1, 1], shape (in_features, out_features).
               C++ normalises [algoWeightMin, algoWeightMax] → cells.
  Input CSV:   unsigned integers, shape (in_features, num_input_vectors).
               One ROW per input element — C++ indexes rows by feature.
               C++ binary-expands each integer to numBitInput bits.
"""

import os, sys, types, subprocess, numpy as np, torch

from pytorch_quantization import cim
from pytorch_quantization.tensor_quant import QuantDescriptor


def make_cim_args(**ov):
    d = dict(
        input_precision=8, weight_precision=8, adc_precision=7,
        dac_precision=1, bitcell=1, sub_array=[128,128], parallel_read=128,
        mem_type="resistive", off_state=6e-3, on_state=6e-3*17,
        mem_states_file="", read_noise=0.0, output_noise=0.0,
        output_noise_file="", vdd=1.0, hardware=1, t=1, v=0.0,
        detect=0, target=0.0, rate_stuck_0=0.0, rate_stuck_1=0.0,
        model="ppa_test", batch_size=1, fake_quant=True,
        write_network=False, hook=False, quant_mode="iw",
        name="test_linear", logger=None, calib=False, ppa=1,
    )
    d.update(ov)
    return types.SimpleNamespace(**d)


def run_ppa_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    if not os.path.isfile('./NeuroSIM/main'):
        print("ERROR: C++ backend not compiled.  cd NeuroSIM && make && cd ..")
        sys.exit(1)

    IN_FEATURES  = 128
    OUT_FEATURES = 64
    args = make_cim_args(batch_size=1)

    # ── 1. Set up CIMLinear ───────────────────────────────────────
    cim.CIMLinear.set_default_quant_desc_input(QuantDescriptor(num_bits=8, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_weight(QuantDescriptor(num_bits=8, axis=0, fake_quant=True))
    cim.CIMLinear.set_default_quant_desc_adc(QuantDescriptor(num_bits=7, fake_quant=True))
    cim.CIMLinear.set_default_cim_args(args)

    layer = cim.CIMLinear(IN_FEATURES, OUT_FEATURES, bias=False).to(device)

    # ── 2. Calibrate and verify forward pass ──────────────────────
    torch.manual_seed(42)
    x = torch.randn(1, IN_FEATURES, device=device)

    layer._input_quantizer.enable()
    layer._weight_quantizer.enable()
    layer._adc_quantizer.disable()
    layer._cim_args.quant_mode = 'iw'
    layer._input_quantizer.amax  = x.abs().max().item()
    layer._weight_quantizer.amax = layer.weight.abs().amax(dim=1, keepdim=True).to(device)

    out = layer(x)
    print(f"Forward pass OK: output shape = {tuple(out.shape)}")

    # ── 3. Write NetWork CSV ──────────────────────────────────────
    record_dir = f'./layer_record_{args.model}'
    os.makedirs(record_dir, exist_ok=True)

    net_csv = f'./NeuroSIM/NetWork_{args.model}.csv'
    with open(net_csv, 'w') as f:
        f.write(f'1,1,{IN_FEATURES},1,1,{OUT_FEATURES},0,1\n')
    print(f"\nNetwork: {net_csv}  →  Linear({IN_FEATURES} → {OUT_FEATURES})")

    # ── 4. Write trace_command.sh header ──────────────────────────
    trace_cmd = os.path.join(record_dir, 'trace_command.sh')
    with open(trace_cmd, 'w') as f:
        f.write(f'./NeuroSIM/main ./NeuroSIM/NetWork_{args.model}.csv '
                f'{args.weight_precision} {args.input_precision} '
                f'{args.sub_array[0]} {args.parallel_read} ')

    # ── 5. Generate trace files ───────────────────────────────────
    # --- Weights: float in [-1, 1], shape (in_features, out_features) ---
    quant_weight = layer._weight_quantizer(layer.weight)   # (out, in)
    w = quant_weight.t()                                    # (in, out)
    w_max = w.abs().max()
    if w_max > 0:
        w = w / w_max                                       # normalise to [-1, 1]

    # --- Inputs: unsigned int, shape (in_features, num_vectors) ---
    #     One ROW per input element — the C++ CopyInput indexes rows by feature.
    #     For a FC layer, num_input_vectors = 1, so shape is (in_features, 1).
    quant_input = layer._input_quantizer(x)                 # (1, in_features)
    i_amax = layer._input_quantizer.amax
    i_bound = (2.0 ** (args.input_precision - 1)) - 1.0
    inp_int = torch.round(quant_input * (i_bound / i_amax)).int()
    inp_int = inp_int + int(2 ** (args.input_precision - 1))   # shift to unsigned
    inp_int = inp_int.clamp(0, int(2**args.input_precision - 1))
    inp_int = inp_int.squeeze(0).unsqueeze(1)               # (in_features, 1)

    weight_file = os.path.join(record_dir, f'weight{args.name}.csv')
    input_file  = os.path.join(record_dir, f'input{args.name}.csv')

    np.savetxt(weight_file, w.cpu().detach().numpy(), delimiter=",", fmt='%10.5f')
    np.savetxt(input_file,  inp_int.cpu().detach().numpy(), delimiter=",", fmt='%d')

    with open(trace_cmd, 'a') as f:
        f.write(f'{weight_file} {input_file} ')

    print(f"\nTraces:")
    print(f"  {weight_file}  shape={tuple(w.shape)}")
    print(f"  {input_file}   shape={tuple(inp_int.shape)}")

    # ── 6. Run C++ PPA ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running NeuroSIM C++ PPA estimation...")
    print(f"{'='*60}\n")

    result = subprocess.run(['/bin/bash', trace_cmd], capture_output=True, text=True)

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("\nSTDOUT:\n", result.stdout[:3000] if result.stdout else "(empty)")
        print("\nC++ backend failed!")
        sys.exit(1)

    print(result.stdout)

    if any(k in result.stdout for k in ["Total", "Energy", "Latency"]):
        print(f"\n{'='*60}")
        print("PPA estimation completed successfully!")
        print(f"{'='*60}")
    else:
        print("\nWARNING: unexpected output — check NeuroSIM/Param.cpp")


if __name__ == "__main__":
    run_ppa_test()
