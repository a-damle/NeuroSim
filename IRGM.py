import torch
import numpy as np

class IRGM():
    def __init__(self, A, b):
        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.b = torch.as_tensor(b, dtype=torch.float64)
        
        n = b.shape[0]
        self.x = torch.zeros(n, dtype=torch.float64)
        
        #Diagonal Preconditioner
        self.M_diag = torch.diag(self.A)
        #M_inv_A = M^-1 @ A
        M_inv_A = self.A / self.M_diag.unsqueeze(1)
        
        self.A_tilde = M_inv_A - torch.eye(n, dtype=torch.float64)
        
        # In a real hardware scenario, A_tilde is what you program 
        # into the PCM/CIM array once here.
        
    #Solve Ax = b using FP16 LU factorization and triangular solve
    def solve_initial_fp16(self):
        #torch.linalg.lu_factor doesn't support 'Half' on CPU
        #A_f16 = self.A.to(torch.float16)
        #b_f16 = self.b.to(torch.float16)
        
        A_f32 = self.A.to(torch.float32)
        b_f32 = self.b.to(torch.float32)
        
        #LU Factorization PA = LU, Returns a packed LU tensor and pivot indices
        self.LU, self.pivots = torch.linalg.lu_factor(A_f32)
        
        x_initial = torch.linalg.lu_solve(self.LU, self.pivots, b_f32.unsqueeze(1))
        self.x = x_initial.squeeze(1).to(torch.float64)
        return self.x
        
    def residual(self):
        #return self.A @ self.x - self.b
        return self.b - (self.A @ self.x)
        
    def solve(self, max_outer_iters=10, tol=1e-12):
        self.solve_initial_fp16()
        
        for i in range(max_outer_iters):
            r = self.residual() #residual vector r = b - Ax
            beta = torch.linalg.norm(r) #L2 norm of r
            
            print(f"Outer Iteration {i}: Residual Norm = {beta:.2e}")
            
            if beta < tol:
                print("Convergence")
                break
            
            #c = self.gmres(r)
            c = self.gmres_cim(r)
            self.x = self.x + c
            
    def gmres(self, r, m=20, tol=1e-16):
        n = r.shape[0]
        
        #Preconditioner
        r_f32 = r.to(torch.float32)
        z = torch.linalg.lu_solve(self.LU, self.pivots, r_f32.unsqueeze(1)).squeeze(1)
        z = z.to(torch.float64)
        
        beta = torch.linalg.vector_norm(z)
        
        #beta = torch.linalg.vector_norm(r)
        
        if beta < tol:
            return torch.zeros_like(r)
        
        V = torch.zeros((n, m+1), dtype=torch.float64) #m+1 to avoid index error if m is 0
        H = torch.zeros((m + 1, m), dtype=torch.float64)
        
        #V[:, 0] = r / beta
        V[:, 0] = z / beta
        
        print(f"\n--- Starting GMRES ---")
        print(f"Initial beta (Residual Norm): {beta:.4e}")
        
        m_final = m
        
        for k in range(m):
            print(f"  Iteration {k}: V[:, {k}] norm: {torch.linalg.vector_norm(V[:, k]):.4f}")
            
            Av = self.A @ V[:, k] #switch to ACIM multiplication?
            Av_f32 = Av.to(torch.float32) 
            
            w = torch.linalg.lu_solve(self.LU, self.pivots, Av_f32.unsqueeze(1)).squeeze(1)
            w = w.to(torch.float64)
            
            for l in range(k + 1):
                H[l, k] = torch.linalg.vecdot(w, V[:, l])
                w = w - H[l, k] * V[:, l]
                
            h_next_k = torch.linalg.vector_norm(w)
            H[k + 1, k] = h_next_k
            
            if h_next_k < tol:
                print(f"  Inner convergence reached at k={k}")
                m_final = k + 1
                break
                
            V[:, k + 1] = w / h_next_k
            
        # Solve least squares AFTER Arnoldi loop
        e1 = torch.zeros(m_final + 1, dtype=torch.float64)
        e1[0] = 1.0
        target = beta * e1

        y = torch.linalg.lstsq(H[:m_final+1, :m_final], target).solution
        c = V[:, :m_final] @ y
        
        print(f"Final correction norm: {torch.linalg.vector_norm(c):.4e}\n")
        
        return c
        
    def gmres_cim(self, r, m=20, tol=1e-16):
        n = r.shape[0]
        b_inner = r / self.M_diag
        beta = torch.linalg.vector_norm(b_inner)
        
        if beta < tol:
            return torch.zeros_like(r)
            
        V = torch.zeros((n, m+1), dtype=torch.float64) #m+1 to avoid index error if m is 0
        H = torch.zeros((m + 1, m), dtype=torch.float64)
        V[:, 0] = b_inner / beta
        
        m_final = m
        
        for k in range(m):
            v_k = V[:, k]
            
            # This A_tilde @ v_k is the ONLY operation that would happen on the CIM hardware.
            w_cim = self.A_tilde @ v_k 
            w = v_k + w_cim  # Digital addition
            
            # Modified Gram-Schmidt (Digital)
            for l in range(k + 1):
                H[l, k] = torch.linalg.vecdot(w, V[:, l])
                w = w - H[l, k] * V[:, l]
                
            h_next_k = torch.linalg.vector_norm(w)
            H[k + 1, k] = h_next_k
            
            if h_next_k < tol:
                print(f"  Inner convergence reached at k={k}")
                m_final = k + 1
                break
                
            V[:, k + 1] = w / h_next_k
            
        # Solve least squares AFTER Arnoldi loop
        e1 = torch.zeros(m_final + 1, dtype=torch.float64)
        e1[0] = 1.0
        target = beta * e1

        y = torch.linalg.lstsq(H[:m_final+1, :m_final], target).solution
        c = V[:, :m_final] @ y
        
        print(f"Final correction norm: {torch.linalg.vector_norm(c):.4e}\n")
        
        return c

# Test case 
torch.manual_seed(0)
n = 10
# Create orthogonal matrix
Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
sing_vals = torch.logspace(0, -6, n, dtype=torch.float64)
A = Q @ torch.diag(sing_vals) @ Q.T
b = torch.randn(n, dtype=torch.float64)

#Solver
solver = IRGM(A, b)
solver.solve()
x_irgm = solver.x

# Gold standard
x_ref = torch.linalg.solve(A, b)

#Compare IRGM vs standard solver
err = torch.linalg.norm(x_irgm - x_ref)
print("Solution error:", err)
rel_err = err / torch.linalg.norm(x_ref)
print("Relative error:", rel_err)
res_igrm = torch.linalg.norm(b - A @ x_irgm)
res_ref = torch.linalg.norm(b - A @ x_ref)
print(res_igrm, res_ref)