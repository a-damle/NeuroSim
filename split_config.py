"""
split_config.py — Generate one config file per individual run.

Each config has exactly one (type, w_bits, i_bits, parallelism, K) combo.
Used with xargs -P for perfectly balanced parallel execution.

Usage:
    python split_config.py sweep_config.json

Creates: jobs/job_t1_w8_i8_p128_k1.json, etc.
Outputs: jobs/job_list.txt with one filename per line
"""

import json
import os
import sys

config_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_config.json"

with open(config_path) as f:
    cfg = json.load(f)

os.makedirs("jobs", exist_ok=True)

K_values = cfg.get("K_values", [1])
row_par = cfg["row_parallelism"]
job_list = []

for mtype, scfg in cfg["sweep_configs"].items():
    for wb in scfg["w_bits"]:
        for ib in scfg["i_bits"]:
            for pr in row_par:
                for K in K_values:
                    job_cfg = dict(cfg)
                    job_cfg["sweep_configs"] = {
                        mtype: {
                            "name": scfg["name"],
                            "precond": scfg["precond"],
                            "w_bits": [wb],
                            "i_bits": [ib],
                        }
                    }
                    job_cfg["row_parallelism"] = [pr]
                    job_cfg["K_values"] = [K]

                    tag = f"t{mtype}_w{wb}_i{ib}_p{pr}_k{K}"
                    filename = f"jobs/job_{tag}.json"

                    with open(filename, 'w') as f:
                        json.dump(job_cfg, f, indent=2)

                    job_list.append(tag)

# Write job list
with open("jobs/job_list.txt", 'w') as f:
    for tag in job_list:
        f.write(tag + '\n')

print(f"Generated {len(job_list)} job configs in jobs/")
