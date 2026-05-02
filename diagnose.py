"""Diagnose why types 4 and 6 fail — check diagonal entries."""
import torch
from test_matrices import TestMatrixGenerator

gen = TestMatrixGenerator(100, seed=42)

for mtype in [1, 2, 3, 4, 5, 6]:
    fn = {1: gen.type1, 2: lambda: gen.type2(1e4), 3: lambda: gen.type3(1e4),
          4: lambda: gen.type4(1e4), 5: lambda: gen.type5(1e4), 6: lambda: gen.type6(1e4)}
    A = fn[mtype]()
    d = torch.diag(A)
    print(f"Type {mtype}:  diag min={d.abs().min():.4e}  max={d.abs().max():.4e}  "
          f"ratio={d.abs().max()/d.abs().min():.1e}  "
          f"near-zero (<1e-2): {(d.abs() < 1e-2).sum().item()}")
