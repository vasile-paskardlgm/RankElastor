import torch
import torch.nn as nn
import torch.nn.functional as F

# Global Permutation
class GlobalPermutation(nn.Module):
    def __init__(self, T, D, H, mode='soft_row', hard_temp=0.7, sinkhorn_iter=20):
        """
        Args:
            T: Number of rows (Token number)
            D: Input dimension
            H: Number of heads (subvector splits)
            mode: 'soft_row', 'hard_row', 'soft_row_col', 'hard_row_col'
            hard_temp: Temperature for Gumbel-Softmax (lower -> harder)
            sinkhorn_iter: Iterations for doubly-stochastic normalization
        """
        super().__init__()
        assert D % H == 0, "D must be divisible by H"
        
        self.T = T
        self.D = D
        self.H = H
        self.sub_dim = D // H
        self.mode = mode
        self.temp = hard_temp
        self.sinkhorn_iter = sinkhorn_iter
        
        # Learnable Weights for Row Permutation P (T x T)
        # We initialize with a slight noise around Identity to aid convergence
        self.row_logits = nn.Parameter(torch.eye(T) + torch.randn(T, T) * 0.01)
        
        # Learnable Weights for Column Permutation Q (H x H)
        if 'col' in mode:
            self.col_logits = nn.Parameter(torch.eye(H) + torch.randn(H, H) * 0.01)
        else:
            self.register_parameter('col_logits', None)

    def sinkhorn(self, logits, n_iter):
        """
        Applies Sinkhorn-Knopp algorithm in log-space.
        Input: (N, N) logits
        Output: (N, N) doubly stochastic matrix
        """
        log_alpha = logits
        for _ in range(n_iter):
            # Row normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Column normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def get_perm_matrix(self, logits):
        """Gumbel-Sinkhorn. We generate P or Q based on mode (Soft vs Nearly Hard)"""
        if 'hard' in self.mode:
            # Sample Gumbel noise Noise must be on the same device as logits
            u = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            
            # Apply temperature (low temp = sharp distribution)
            noisy_logits = (logits + gumbel) / self.temp
            
            # Sinkhorn normalization
            return self.sinkhorn(noisy_logits, self.sinkhorn_iter)
        else:
            # Soft weight permutation
            return self.sinkhorn(logits, self.sinkhorn_iter)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Bs, T, D)
        Returns:
            out: Permuted tensor (Bs, T, D)
            (P, Q): The permutation matrices used (T,T) and (H,H)
        """
        # Ensure input is (Bs, T, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        B, T, D = x.shape
        assert T == self.T and D == self.D, f"Input shape mismatch. Expected (B, {self.T}, {self.D}), got {x.shape}"
        
        # Reshape to view subvectors: (Bs, T, H, D/H)
        # Let 'd' be the subvector dimension
        x_view = x.view(B, T, self.H, self.sub_dim) 
        
        # Get Row Permutation Matrix P (T x T)
        P = self.get_perm_matrix(self.row_logits)
        
        # Apply Row Permutation
        # Equation: Output[b, t, h, d] = sum_k ( P[t, k] * Input[b, k, h, d] )
        # Einsum: 'tk' (P) and 'bkhd' (Input) -> 'bthd'
        x_perm = torch.einsum('tk, bkhd -> bthd', P, x_view)
        
        Q = None
        # Apply Column Permutation (if active)
        if 'col' in self.mode:
            # Get Col Permutation Matrix Q (H x H)
            Q = self.get_perm_matrix(self.col_logits)
            
            # Equation: Output[b, t, h, d] = sum_l ( Q[h, l] * Input[b, t, l, d] )
            # Einsum: 'hl' (Q) and 'btld' (Input) -> 'bthd'
            x_perm = torch.einsum('hl, btld -> bthd', Q, x_perm)
            
        # Flatten back to (Bs, T, D)
        out = x_perm.reshape(B, T, D)
        
        return out

r''''
# ==========================================
# Batch Demonstration
# ==========================================

# Settings
Bs = 8       # Batch size
T = 10       # Time steps (Rows)
D = 32       # Feature dim
H = 4        # Heads (Col divisions)
input_tensor = torch.randn(Bs, T, D)

print(f"Input shape: {input_tensor.shape}")

# --- Approach 1: Soft Row ---
model1 = GlobalPermutation(T, D, H, mode='soft_row')
out1, (P1, _) = model1(input_tensor)
print(f"\n1. Soft Row Output: {out1.shape}")
print(f"   P matrix shape: {P1.shape} (Broadcasts to all {Bs} samples)")

# --- Approach 4: Nearly Hard Row & Column ---
# hard_temp=0.1 makes it very close to discrete 0/1
model4 = GlobalPermutation(T, D, H, mode='hard_row_col', hard_temp=0.01)
out4, (P4, Q4) = model4(input_tensor)

print(f"\n4. Hard Row+Col Output: {out4.shape}")
print("   P sample (Top-left 4x4):\n", P4[:4,:4].detach().numpy().round(2))
print("   Q sample (Full HxH):\n", Q4.detach().numpy().round(2))

# Verification of 'Hard' properties
print(f"   Is P close to binary? {((P4 > 0.99) | (P4 < 0.01)).all().item()}")
'''


# Adaptive Permutation
class DataAdaptivePermutation(nn.Module):
    def __init__(self, T, D, H, mode='soft_row', hard_temp=0.7, sinkhorn_iter=20):
        super().__init__()
        assert D % H == 0
        
        self.T = T
        self.D = D
        self.H = H
        self.sub_dim = D // H
        self.mode = mode
        self.temp = hard_temp
        self.sinkhorn_iter = sinkhorn_iter
        
        # Row Permutation Predictor
        self.row_predictor = NeuralSortPredictor(D, T)
        
        # Col Permutation Predictor
        if 'col' in mode:
            self.col_predictor = NeuralSortPredictor(self.sub_dim, H)
        
    def sinkhorn(self, logits):
        """Standard Sinkhorn for (B, N, N)"""
        log_alpha = logits
        for _ in range(self.sinkhorn_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def gumbel_sinkhorn(self, logits):
        """Gumbel-Sinkhorn for (B, N, N) - Nearly Hard"""
        # Sample noise same shape as logits (B, N, N)
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        
        noisy_logits = (logits + gumbel) / self.temp
        return self.sinkhorn(noisy_logits)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        B, T, D = x.shape
        
        # Generate Row Permutation P (Sample Specific)
        # Predict logits from data: (B, T, T)
        row_logits = self.row_predictor(x)
        
        if 'hard' in self.mode:
            P = self.gumbel_sinkhorn(row_logits)
        else:
            P = self.sinkhorn(row_logits)
            
        # Reshape Input
        x_view = x.view(B, T, self.H, self.sub_dim) # (B, T, H, d)
        
        # Apply Row Permutation
        # P is (B, T, T), x is (B, T, H, d)
        x_perm = torch.matmul(P, x_view.reshape(B, T, -1)) # (B, T, H*d)
        x_perm = x_perm.view(B, T, self.H, self.sub_dim)   # Back to (B, T, H, d)
        
        Q = None
        # Generate & Apply Col Permutation
        if 'col' in self.mode:
            # (B, T, H, d) -> mean over T -> (B, H, d)
            head_features = x_view.mean(dim=1) 
            
            col_logits = self.col_predictor(head_features) # (B, H, H)
            
            if 'hard' in self.mode:
                Q = self.gumbel_sinkhorn(col_logits)
            else:
                Q = self.sinkhorn(col_logits)
            
            # Apply Q: (B, H, H) @ (B, T, H, d)
            x_perm = torch.einsum('bhl, bthd -> btld', Q, x_perm) # Output: (B, T, H, d)
            
        out = x_perm.reshape(B, T, D)

        return out

class NeuralSortPredictor(nn.Module):
    def __init__(self, in_dim, num_elements):
        super().__init__()
        self.scorer = nn.Linear(in_dim, 1)
        
        # Fixed "Target" values for the columns (The positions)
        # Position 0 has weight T, Position T has weight 0
        self.register_buffer(
            'position_weights', 
            torch.arange(num_elements, 0, -1).float().view(1, 1, num_elements) 
        ) # (1, 1, T)
        
        # Learnable Bias to force Identity Initialization
        # Row 0 to match with Col 0 (which has high position_weight).
        self.init_bias = nn.Parameter(
             torch.arange(num_elements, 0, -1).float().view(1, num_elements, 1) * 2.0
        ) # (1, T, 1)

    def forward(self, x):
        # x: (B, T, D)
        
        # Predict Data-Dependent Scores
        raw_scores = self.scorer(x) # (B, T, 1)
        
        # Add Bias (Initializes to Identity Order)
        # As training proceeds, 'raw_scores' (data dependent) should overwhelm 'init_bias'
        total_scores = raw_scores + self.init_bias
        
        # Create Logits for Sinkhorn
        # Matches High Scores to Early Positions
        logits = torch.matmul(total_scores, self.position_weights) # (B, T, T)
        
        return logits


r'''
# ----------------------------------------------------------------
# Usage Example
# ----------------------------------------------------------------
Bs, T, D, H = 2, 5, 20, 4
x = torch.randn(Bs, T, D)

# Define Data-Adaptive Model (Hard Row & Col)
model = DataAdaptivePermutation(T, D, H, mode='hard_row_col', hard_temp=0.05)
out, (P, Q) = model(x)

print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print(f"P shape: {P.shape} (Batch-specific!)")
print(f"Q shape: {Q.shape} (Batch-specific!)")

print("\n--- Check Data Adaptation ---")
diff = (P[0] - P[1]).abs().sum()
print(f"Difference between P for batch item 0 and 1: {diff.item():.4f}")
if diff > 0.1:
    print(">> Success: Permutation matrices are different for different inputs.")
else:
    print(">> Warning: Permutations look identical (weights might be too small or identity bias too high).")
'''