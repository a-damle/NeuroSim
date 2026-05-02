"""
merge_results.py — Combine per-type result files into one.

Usage:
    python merge_results.py results/exp12_rram_type1_results.pt results/exp12_rram_type2_results.pt ... -o results/exp12_rram_results.pt
"""

import argparse
import torch


def merge(input_paths, output_path):
    merged_results = {}
    merged_baselines = {}
    config = None
    device_type = None
    device_params = None

    for path in input_paths:
        d = torch.load(path, weights_only=False, map_location='cpu')
        merged_results.update(d["results"])
        if "baselines" in d:
            merged_baselines.update(d["baselines"])

        if config is None:
            config = d.get("config")
            device_type = d.get("device_type")
            device_params = d.get("device_params")

    torch.save({
        "results": merged_results,
        "baselines": merged_baselines,
        "config": config,
        "device_type": device_type,
        "device_params": device_params,
    }, output_path)

    print(f"Merged {len(input_paths)} files → {output_path}")
    print(f"Total configs: {len(merged_results)}")
    print(f"Total baselines: {len(merged_baselines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input .pt files")
    parser.add_argument("-o", "--output", required=True, help="Output .pt file")
    args = parser.parse_args()

    merge(args.inputs, args.output)
