import torch
import numpy

class TestMatrixGenerator:
    def __init__(self, matrix_size):
        self.n = matrix_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Random numbers with diagonal modified to be dominant
    def type1(self):
        #random n x n matrix
        matrix1, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        #diagonally dominant: for every row, the magnitude of the diagonal entry 
        # >= sum of magnitude of all other non-diagonal entries in same row.
        
        for i in range(self.n):
            #Just add the sum of the row to the diagonal entry for every row, making it strictly dominant
            matrix1[i, i] += torch.sum(torch.abs(matrix1[i]))
        
        return matrix1
        
    #Positive eigen value, Random sigma between 1/cond and 1 such that logarithms are uniformly distributed
    def type2(self, cond=100.0):
        Q, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        log_min = -torch.log(torch.tensor(cond, dtype=torch.float64, device=self.device))  #log(1/cond) = -log(cond)
        log_max = torch.tensor(0.0, dtype=torch.float64, device=self.device) #log(1) = 0
        log = log_min + (log_max - log_min) * torch.rand(self.n, dtype=torch.float64, device=self.device)
        
        singular_values = torch.exp(log)
        Sigma  = torch.diag(singular_values)

        matrix2 = Q @ Sigma @ Q.T
        return matrix2
    
    #Positive eigen value, with clustered singular values [1,...,1,1/cond]
    def type3(self, cond=100.0):
        Q, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        singular_values  = torch.ones(self.n, dtype=torch.float64, device=self.device)
        singular_values[-1] = 1.0 / cond
        Sigma  = torch.diag(singular_values)
        
        matrix3 = Q @ Sigma @ Q.T
        return matrix3
    
    #Clustered singular values (1,...,1,1/cond)
    def type4(self, cond=100.0):
        U, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        V, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        singular_values  = torch.ones(self.n, dtype=torch.float64, device=self.device)
        singular_values[-1] = 1.0 / cond
        Sigma  = torch.diag(singular_values)
        
        matrix4 = U @ Sigma @ V.T
        return matrix4
    
    #Positive eigen value. Arithmetic distribution of singular values 1 - (i-1/n-1)(1-1/cond)
    def type5(self, cond=100.0):
        Q, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        if self.n == 1:
            singular_values = torch.tensor([1.0 / cond], dtype=torch.float64, device=self.device)
        else:
            i = torch.arange(self.n, dtype=torch.float64, device=self.device)
            singular_values = 1.0 - (i / (self.n - 1)) * (1.0 - 1.0 / cond)
            
        Sigma  = torch.diag(singular_values)
        
        matrix5 = Q @ Sigma @ Q.T
        return matrix5
        
    # Arithmetic distribution of singular values 1 - (i-1/n-1)(1-1/cond)
    def type6(self, cond=100.0):
        U, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        V, _ = torch.linalg.qr(torch.randn(self.n, self.n, dtype=torch.float64, device=self.device))
        
        if self.n == 1:
            singular_values = torch.tensor([1.0 / cond], dtype=torch.float64, device=self.device)
        else:
            i = torch.arange(self.n, dtype=torch.float64, device=self.device)
            singular_values = 1.0 - (i / (self.n - 1)) * (1.0 - 1.0 / cond)
            
        Sigma  = torch.diag(singular_values)
        
        matrix6 = U @ Sigma @ V.T
        return matrix6