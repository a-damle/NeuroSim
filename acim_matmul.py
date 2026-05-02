"""
acim_matmul.py — Quantized matrix multiplication matching NeuroSim v1.5.

Standalone module with NO NeuroSim dependencies — only PyTorch.
"""

import torch


def fake_quant(x: torch.Tensor, amax: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Fake-quantize: scale → round → clamp → descale."""
    max_bound = (2.0 ** (num_bits - 1)) - 1.0
    min_bound = -max_bound

    amax = amax.float()
    scale = torch.where(amax > 0, max_bound / amax, torch.zeros_like(amax))

    quantized = (x.float() * scale).round_().clamp_(min_bound, max_bound)
    return (quantized / scale.clamp(min=1e-38)).to(x.dtype)


def acim_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    weight_bits: int = 8,
    input_bits: int = 8,
) -> torch.Tensor:
    """Quantized matrix multiplication: C = quant(A) @ quant(B).

    Args:
        A:            Weight matrix, shape (m, k). Per-row quantization.
        B:            Input matrix,  shape (k, n). Per-tensor quantization.
        weight_bits:  Bit-width for A.
        input_bits:   Bit-width for B.

    Returns:
        C: shape (m, n).
    """
    amax_A = A.abs().amax(dim=1, keepdim=True)
    A_q = fake_quant(A, amax_A, weight_bits)

    amax_B = B.abs().max()
    B_q = fake_quant(B, amax_B, input_bits)

    return A_q @ B_q
