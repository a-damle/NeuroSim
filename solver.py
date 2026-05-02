"""
solver.py — Mixed-precision iterative refinement with unpreconditioned GMRES.

Based on Le Gallo et al. (2018), Algorithm 2.

Precision: all arithmetic is FP64. The only low-precision operation is the
pluggable matmul_fn for the CIM array simulation.

Two preconditioning modes:
  precond='diag': Stores A_tilde = M^{-1}A - I (diagonal removed).
  precond=None:   Stores A directly.
"""

import torch


def _default_matmul(A, x):
    """FP64 baseline matmul."""
    return A @ x


class MixedPrecisionSolver:
    def __init__(self, A, b, matmul_fn=None, precond='diag'):
        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.b = torch.as_tensor(b, dtype=torch.float64)
        self.device = self.A.device
        self.n = b.shape[0]
        self.matmul_fn = matmul_fn or _default_matmul
        self.precond = precond

        if precond == 'diag':
            self.M_diag = torch.diag(self.A)

            row_norms = self.A.abs().sum(dim=1)
            ratio = self.M_diag.abs() / row_norms
            if ratio.min() < 1e-3:
                print(f"  WARNING: diag(A) has entries much smaller than row norms "
                      f"(min ratio = {ratio.min():.2e}). Consider precond=None.")

            M_inv_A = self.A / self.M_diag.unsqueeze(1)
            self.W_stationary = M_inv_A - torch.eye(self.n, dtype=torch.float64, device=self.device)
        else:
            self.M_diag = None
            self.W_stationary = self.A.clone()

    def solve(self, max_outer=50, inner_iters=20, tol=1e-12, verbose=True,
              diverge_factor=100.0):
        """Run iterative refinement.

        Early stopping:
          - Converged:  ||r|| < tol
          - Diverging:  ||r|| > diverge_factor × ||r_0||
        """
        x = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        history = []
        stop_reason = "max_iters"

        for i in range(max_outer):
            r = self.b - self.A @ x
            res_norm = torch.linalg.vector_norm(r).item()
            history.append(res_norm)

            if verbose:
                print(f"  Outer {i:3d}: ||r|| = {res_norm:.4e}")

            # Convergence
            if res_norm < tol:
                stop_reason = "converged"
                if verbose:
                    print(f"  Converged at outer iteration {i}")
                break

            # Divergence: residual much worse than initial
            if i > 0 and res_norm > diverge_factor * history[0]:
                stop_reason = "diverged"
                if verbose:
                    print(f"  Diverged at outer iteration {i} "
                          f"(||r|| = {res_norm:.1e} > {diverge_factor}× initial)")
                break

            z = self._gmres_inner(r, m=inner_iters)
            x = x + z

        self.x = x
        self.stop_reason = stop_reason
        return x, history

    def _gmres_inner(self, r, m=20):
        if self.precond == 'diag':
            b_inner = r / self.M_diag
        else:
            b_inner = r

        beta = torch.linalg.vector_norm(b_inner).item()
        if beta < 1e-16:
            return torch.zeros(self.n, dtype=torch.float64, device=self.device)

        V = torch.zeros((self.n, m + 1), dtype=torch.float64, device=self.device)
        H = torch.zeros((m + 1, m), dtype=torch.float64, device=self.device)
        V[:, 0] = b_inner / beta

        m_actual = m

        for k in range(m):
            v_k = V[:, k]

            # === CIM OPERATION ===
            w_cim = self.matmul_fn(self.W_stationary, v_k.unsqueeze(1)).squeeze(1)

            if self.precond == 'diag':
                w = v_k + w_cim
            else:
                w = w_cim

            # Modified Gram-Schmidt (FP64)
            for l in range(k + 1):
                H[l, k] = torch.linalg.vecdot(w, V[:, l])
                w = w - H[l, k] * V[:, l]

            h_next = torch.linalg.vector_norm(w).item()
            H[k + 1, k] = h_next

            if h_next < 1e-16:
                m_actual = k + 1
                break

            V[:, k + 1] = w / h_next

        e1 = torch.zeros(m_actual + 1, dtype=torch.float64, device=self.device)
        e1[0] = beta
        y = torch.linalg.lstsq(H[:m_actual + 1, :m_actual], e1).solution

        return V[:, :m_actual] @ y
