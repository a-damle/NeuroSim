"""
test_matrices.py — Generate the 6 test matrix types from Haidar et al. Table II.

Types 1, 2, 3, 5: symmetric (Q Σ Qᵀ) → positive eigenvalues
Types 4, 6:        non-symmetric (U Σ Vᵀ) → eigenvalues may be complex
"""

import torch


class TestMatrixGenerator:
    def __init__(self, n, seed=42, device=None):
        self.n = n
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        torch.manual_seed(self.seed)

    def _randn(self, *shape):
        return torch.randn(*shape, dtype=torch.float64, device=self.device)

    def type1(self):
        """Diagonally dominant."""
        self._set_seed()
        A, _ = torch.linalg.qr(self._randn(self.n, self.n))
        for i in range(self.n):
            A[i, i] += torch.sum(torch.abs(A[i]))
        return A

    def type2(self, cond=1e4):
        """Positive λ, log-uniform singular values in [1/cond, 1]."""
        self._set_seed()
        Q, _ = torch.linalg.qr(self._randn(self.n, self.n))
        log_sv = -torch.log(torch.tensor(cond, device=self.device)) * \
                  torch.rand(self.n, dtype=torch.float64, device=self.device)
        return Q @ torch.diag(torch.exp(log_sv)) @ Q.T

    def type3(self, cond=1e4):
        """Positive λ, clustered singular values [1,...,1, 1/cond]."""
        self._set_seed()
        Q, _ = torch.linalg.qr(self._randn(self.n, self.n))
        sv = torch.ones(self.n, dtype=torch.float64, device=self.device)
        sv[-1] = 1.0 / cond
        return Q @ torch.diag(sv) @ Q.T

    def type4(self, cond=1e4):
        """Clustered singular values, general eigenvalues (non-symmetric)."""
        self._set_seed()
        U, _ = torch.linalg.qr(self._randn(self.n, self.n))
        V, _ = torch.linalg.qr(self._randn(self.n, self.n))
        sv = torch.ones(self.n, dtype=torch.float64, device=self.device)
        sv[-1] = 1.0 / cond
        return U @ torch.diag(sv) @ V.T

    def type5(self, cond=1e4):
        """Positive λ, arithmetic singular values."""
        self._set_seed()
        Q, _ = torch.linalg.qr(self._randn(self.n, self.n))
        i = torch.arange(self.n, dtype=torch.float64, device=self.device)
        sv = 1.0 - (i / (self.n - 1)) * (1.0 - 1.0 / cond)
        return Q @ torch.diag(sv) @ Q.T

    def type6(self, cond=1e4):
        """Arithmetic singular values, general eigenvalues (non-symmetric)."""
        self._set_seed()
        U, _ = torch.linalg.qr(self._randn(self.n, self.n))
        V, _ = torch.linalg.qr(self._randn(self.n, self.n))
        i = torch.arange(self.n, dtype=torch.float64, device=self.device)
        sv = 1.0 - (i / (self.n - 1)) * (1.0 - 1.0 / cond)
        return U @ torch.diag(sv) @ V.T

    def all_types(self, cond=1e4):
        return {
            "Type 1 (diag dominant)":    self.type1(),
            "Type 2 (log-uniform σ)":    self.type2(cond),
            "Type 3 (clustered σ, +λ)":  self.type3(cond),
            "Type 4 (clustered σ)":      self.type4(cond),
            "Type 5 (arithmetic σ, +λ)": self.type5(cond),
            "Type 6 (arithmetic σ)":     self.type6(cond),
        }
