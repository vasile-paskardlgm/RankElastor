import torch
import torch.nn as nn
import torch.nn.functional as F
from .Permutation import GlobalPermutation, DataAdaptivePermutation
from .utils import GRN, LRFlaMix, RMSNorm

class Token_Mixing(nn.Module):
    def __init__(self,
                 T: int,
                 D: int,
                 use_opq: bool = False,
                 opq_momentum: float = 0.01,
                 mixing_type: int = 2,
                 mlp_params: dict = None,
                 attention_params: dict = None):
        """
        Args:
            T: Number of tokens.
            D: Embedding dimension.
            use_opq: Whether to use Parametric OPQ (Eigenvalue Allocation).
            opq_momentum: How fast the covariance matrix updates.
                    Keep small (0.01 - 0.001) for stable rotation updates.
            mixing_type:
                1: Identity
                2: Non-Parametric (Permutation), rankmixer analogy
                999: Flattened Mixing (Analogy of CrossInteraction)
                etc...
            mlp_params: Config for any Types applying MLP modules.
            attention_params: Config for any Types applying attention modules.
        """
        super(Token_Mixing, self).__init__()

        self.T = T
        self.D = D
        self.H = T # H subspaces

        self.use_opq = use_opq
        self.mixing_type = mixing_type
        self.opq_momentum = opq_momentum

        # if D % self.H != 0:
        #     raise ValueError(f"Error: T ({T}) cannot divide D ({D}).")

        # Parametric OPQ Setup (Eigenvalue Allocation)
        if self.use_opq:
            self.register_buffer('opq_R', torch.eye(D))
            self.register_buffer('running_mean', torch.zeros(D))
            self.register_buffer('running_cov', torch.eye(D))
            self.opq_initialized = False

        # Mixing Layers Setup
        def get_mlp_defaults():
            return {'hidden_units': [D*4], 'activation': 'gelu', 'dropout': 0.1, 'use_layer_norm': False}
        def get_attn_defaults():
            return {'dropout': 0.0, 'bias': True}

        if mlp_params is None: mlp_params = get_mlp_defaults()
        if attention_params is None: attention_params = get_attn_defaults()

        # Construction
        if self.mixing_type == 1:
            # Identity
            pass

        elif self.mixing_type == 2:
            # Non-Parametric
            self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type == 3:
        #     # Pure Linear Mixing
        #     # Linear transform on Features (D) then Tokens (T)
        #     # No activation, simple linear projection.
        #     self.linear_col = nn.Linear(D, D)
        #     self.linear_row = nn.Linear(T, T)
        #     self.layer_norm_1 = nn.LayerNorm(D)
        #     self.layer_norm_2 = nn.LayerNorm(D)

        # elif self.mixing_type == 4:
        #     # Parameterized MLP Mixing
        #     self.mlp_col = self._build_mlp(D, D, mlp_params)
        #     self.mlp_row = self._build_mlp(T, T, mlp_params)
        #     self.layer_norm_1 = nn.LayerNorm(D)
        #     self.layer_norm_2 = nn.LayerNorm(D)

        # elif self.mixing_type == 5:
        #     # Parameterized Attention Mixing
        #     defaults = get_attn_defaults()
        #     self.attention = nn.MultiheadAttention(embed_dim=D,
        #                                            num_heads=self.H,
        #                                            dropout=attention_params.get('dropout', defaults['dropout']),
        #                                            bias=attention_params.get('bias', defaults['bias']),
        #                                            batch_first=True)
        #     self.layer_norm = nn.LayerNorm(D)

        # # Learnable Permutation
        # elif self.mixing_type == 6:
        #     self.permute = GlobalPermutation(self.H, (self.T * self.D // self.H), self.T, 'hard_row')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 7:
        #     self.permute = GlobalPermutation(self.H, (self.T * self.D // self.H), self.T, 'soft_row')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 8:
        #     self.permute = GlobalPermutation(self.H, (self.T * self.D // self.H), self.T, 'hard_row_col')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 9:
        #     self.permute = GlobalPermutation(self.H, (self.T * self.D // self.H), self.T, 'soft_row_col')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 10:
        #     self.permute = DataAdaptivePermutation(self.H, (self.T * self.D // self.H), self.T, 'hard_row')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 11:
        #     self.permute = DataAdaptivePermutation(self.H, (self.T * self.D // self.H), self.T, 'soft_row')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 12:
        #     self.permute = DataAdaptivePermutation(self.H, (self.T * self.D // self.H), self.T, 'hard_row_col')
        #     self.layer_norm = nn.LayerNorm(D)
        # elif self.mixing_type == 13:
        #     self.permute = DataAdaptivePermutation(self.H, (self.T * self.D // self.H), self.T, 'soft_row_col')
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type == 14:
        #     # Mixing with Linear Concatenation
        #     # Concatenation then Linear Transformation
        #     # Post-Norm following RankMixer
        #     self.linear_cat = nn.Linear(2*D, D)
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type == 15:
        #     # Bilinear Pooling
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type in [16, 33]:
        #     # Mixing with Nonlinear Concatenation
        #     # Concatenation then MLP
        #     self.mlp_cat = self._build_mlp(2*D, D, mlp_params)
        #     self.layer_norm = nn.LayerNorm(D)
        
        # elif self.mixing_type in [17, 18, 19, 20, 27]:
        #     r'''NOTE: The best practice 17'''
        #     # Mixing with Linear Concatenation + Pre-Norm (5 Variants)
        #     # Concatenation then Linear Transformation
        #     self.linear_cat = nn.Linear(2*D, D)
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type in [21, 22, 23]:
        #     # Mixing with Linear Concatenation + Pre-Norm + Multi-Views (Rotations 90 and 180)
        #     self.layer_norm = nn.LayerNorm(D)
        #     if self.mixing_type in [21, 22]: # x|rot90 and x|rot180
        #         self.linear_cat = nn.Linear(2*D, D)
        #     elif self.mixing_type == 23: # x|rot90|mixed
        #         self.linear_cat = nn.Linear(3*D, D)

        # elif self.mixing_type in [24, 25]:
        #     # Mixing with Linear Concatenation + Pre-Norm + Multi-Views (2D FFT and others)
        #     self.register_buffer('dct_mat', torch.zeros(self.T, self.T))
        #     self._initialized = False
        #     self.layer_norm = nn.LayerNorm(D)

        #     if self.mixing_type == 24:
        #         self.linear_cat = nn.Linear(2*D, D) # x|FFT
        #     elif self.mixing_type == 25:
        #         self.linear_cat = nn.Linear(3*D, D) # x|FFT|mixed

        # elif self.mixing_type == 26:
        #     # Group-wise Outer Product Mixing
        #     self.layer_norm = nn.LayerNorm(D)
        #     self.gop_dim = D**2 // self.H
        #     self.linear_cat = nn.Linear(self.gop_dim + D, D)

        # elif self.mixing_type in [28, 29, 30]:
        #     # Mixing with Linear Concatenation + Pre-Norm + Symmetric-Skew Decomposition (28) + Low-rank (29)
        #     # Concatenation then Linear Transformation
        #     if self.mixing_type == 28: # Symmetric-Skew Decomposition
        #         self.sym_block = nn.Linear(D, D)
        #         self.skew_block = nn.Linear(D, D)
            
        #     elif self.mixing_type in [29, 30]: # Low-rank
        #         r1 = D // 2
        #         r2 = D // 3

        #         self.sym_block = nn.Sequential(
        #             nn.Linear(D, r1, bias=False),  # Compress S
        #             nn.Linear(r1, D, bias=True)    # Expand S
        #         )
                
        #         self.skew_block = nn.Sequential(
        #             nn.Linear(D, r2, bias=False),  # Compress A
        #             nn.Linear(r2, D, bias=True)    # Expand A
        #         )
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type == 31:
        #     # Mixing with Pre-Norm + Polynomial Mixing
        #     # Concatenation then Linear Transformation
        #     self.layer_norm = nn.LayerNorm(D)
        #     # Initialized Degree
        #     self.linear_d0 = nn.Linear(D, D)
        #     # Degree 1
        #     self.linear_d11 = nn.Linear(D, D, bias=False)
        #     self.linear_d12 = nn.Linear(D, D, bias=False)
        #     # Degree 2
        #     self.linear_d21 = nn.Linear(D, D, bias=False)
        #     self.linear_d22 = nn.Linear(D, D, bias=False)

        # elif self.mixing_type == 32:
        #     # Combing Row and Column-wise Mixing with Pre-Norm
        #     self.linear_cat = nn.Linear(2*D, D)
        #     self.linear_tkn = nn.Linear(T, T)
        #     self.layer_norm = nn.LayerNorm(D)

        # elif self.mixing_type == 34:
        #     # Mixing with Linear Concatenation + Peri-Norm (LN at Mixing before and after)
        #     self.linear_cat = nn.Linear(2*D, D)
        #     self.layer_norm_A = nn.LayerNorm(D)
        #     self.layer_norm_B = nn.LayerNorm(D)

        # elif self.mixing_type == 35:
        #     # Mixing with Linear Concatenation + Global Response Normalization
        #     self.linear_cat = nn.Linear(2*D, D)
        #     self.layer_norm_A = nn.LayerNorm(D)
        #     self.layer_norm_B = GRN(D // self.H)

        # elif self.mixing_type == 777:
        #     # Mixing with Flattening + Low-rank Transformation
        #     # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
        #     self.layer_norm = nn.LayerNorm(D)
        #     self.cross_mixing = LRFlaMix(
        #         in_features=T*D,
        #         low_rank=32,
        #         num_experts=4
        #     )

        elif self.mixing_type == 8881:
            # Mixing with Flattening (Kronecker Simplification)
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            self.d_criteo = 26
            self.permutation_mixing = nn.Linear(T*D//self.d_criteo, T*D//self.d_criteo, bias=False)
            self.l_norm = RMSNorm(T*D)

        elif self.mixing_type == 8882:
            # Mixing with Flattening (Kronecker Simplification)
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            self.d_avazu = 2
            self.permutation_mixing = nn.Linear(T*D//self.d_avazu, T*D//self.d_avazu, bias=False)
            self.l_norm = nn.LayerNorm(D)

        elif self.mixing_type == 999:
            # Mixing with Flattening
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            self.cross_mixing = nn.Linear(T*D, T*D)
            self.pre_norm = RMSNorm(T*D)
            
        else:
            raise ValueError("Invalid mixing type.")

    def _build_mlp(self, input_dim, output_dim, params):
        layers = []
        in_d = input_dim
        # Apply defaults inside builder to be safe
        hidden_units = params.get('hidden_units', [input_dim * 4])
        act_name = params.get('activation', 'gelu').lower()
        dropout = params.get('dropout', 0.1)
        use_ln = params.get('use_layer_norm', False)

        if act_name == 'relu': act_fn = nn.ReLU()
        elif act_name == 'gelu': act_fn = nn.GELU()
        elif act_name == 'tanh': act_fn = nn.Tanh()
        else: act_fn = nn.ReLU()

        for h_dim in hidden_units:
            layers.append(nn.Linear(in_d, h_dim))
            if use_ln: layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn)
            if dropout > 0: layers.append(nn.Dropout(dropout))
            in_d = h_dim

        layers.append(nn.Linear(in_d, output_dim))
        return nn.Sequential(*layers)

    def _greedy_eigenvalue_allocation(self, eig_vals, buckets):
        """
        Helper for Parametric OPQ Allocation, which implements the Greedy Eigenvalue Allocation from He et al.
        1. Optimized for GPU:
          Moves small eigenvalue tensor to CPU for the sequential loop
          to avoid GPU kernel launch overhead, then moves indices back.
        2. Goal:
          Balance the product of eigenvalues in each subspace (bucket).
          Equivalent to balancing the sum of log(eigenvalues).
        3. Args:
            eig_vals: Tensor of sorted eigenvalues (Descending).
            buckets: Number of subspaces (H).
        4. Returns:
            permutation_indices: The reordering index to apply to eigenvectors.
        """
        device = eig_vals.device
        log_vals = torch.log(torch.clamp(eig_vals, min=1e-9)).cpu()
        bucket_sums = torch.zeros(buckets, device='cpu')
        assignments = []

        # Pure CPU loop
        for i, val in enumerate(log_vals):
            min_bucket_idx = torch.argmin(bucket_sums)
            bucket_sums[min_bucket_idx] += val
            assignments.append((i, min_bucket_idx))

        assignments.sort(key=lambda x: x[1])
        perm_indices = [x[0] for x in assignments]

        # Back to original device
        return torch.tensor(perm_indices, device=device, dtype=torch.long)

    def _update_opq_parametric(self, x):
        """
        Update R using Covariance SVD + Eigenvalue Allocation.
        Online update of Parametric OPQ Rotation Matrix.
        1. Update Global Covariance (EMA).
        2. PCA: Eigendecomposition.
        3. Eigenvalue Allocation: Permute PCs to balance variance across subspaces.
        """
        with torch.no_grad():
            x_flat = x.reshape(-1, self.D)
            batch_size = x_flat.shape[0]

            # Update Stats
            batch_mean = x_flat.mean(dim=0)
            x_centered = x_flat - batch_mean
            batch_cov = torch.matmul(x_centered.T, x_centered) / (batch_size - 1)

            if not self.opq_initialized:
                self.running_mean.copy_(batch_mean)
                self.running_cov.copy_(batch_cov)
                self.opq_initialized = True
            else:
                m = self.opq_momentum
                self.running_mean.mul_(1 - m).add_(batch_mean * m)
                self.running_cov.mul_(1 - m).add_(batch_cov * m)

            # SVD (Eigh for symmetric)
            eig_vals, eig_vecs = torch.linalg.eigh(self.running_cov)
            eig_vals = eig_vals.flip(0)
            eig_vecs = eig_vecs.flip(1)

            # Allocation
            perm_indices = self._greedy_eigenvalue_allocation(eig_vals, self.H)
            U_permuted = eig_vecs[:, perm_indices]

            # Update R
            self.opq_R.copy_(U_permuted)

    def get_dct_matrix(N, device):
        """Generates a standard Type-II DCT Matrix."""
        n = torch.arange(N, device=device).view(1, -1)
        k = torch.arange(N, device=device).view(-1, 1)
        # Basis: cos(pi * k * (2n+1) / 2N)
        dct_mat = torch.cos(torch.pi * k * (2 * n + 1) / (2 * N))
        dct_mat[0, :] *= 1.0 / torch.sqrt(torch.tensor(2.0))
        dct_mat *= torch.sqrt(torch.tensor(2.0 / N))
        return dct_mat

    def _init_dct(self, device):
            """Generates the DCT-II basis matrix once."""
            N = self.T
            dtype = self.dct_mat.dtype
            n = torch.arange(N, device=device, dtype=dtype).view(1, -1)
            k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
            
            # Standard DCT-II formula
            mat = torch.cos(torch.pi * k * (2 * n + 1) / (2 * N))
            mat[0, :] *= torch.rsqrt(torch.tensor(2.0, device=device))
            mat *= torch.sqrt(torch.tensor(2.0 / N, device=device)) # This is the "norm=ortho"
            
            self.dct_mat.copy_(mat)
            self._initialized = True

    def forward(self, x):
        is_batched = True
        if x.dim() == 2:
            is_batched = False
            x = x.unsqueeze(0)

        Bs, T, D = x.shape

        # OPQ Preprocessing
        if self.use_opq:
            if self.training:
                self._update_opq_parametric(x)
            x_opq = torch.matmul(x, self.opq_R)
        else:
            x_opq = x

        # Mixing
        if self.mixing_type == 1:
            out = x_opq

        elif self.mixing_type == 2:
            # Non-Parametric
            sub_dim = D // self.H
            x_reshaped = x_opq.view(Bs, T, self.H, sub_dim)
            x_permuted = x_reshaped.permute(0, 2, 1, 3)
            mixed = x_permuted.reshape(Bs, self.H, -1)
            out = self.layer_norm(x_opq + mixed)

        elif self.mixing_type == 8881: # criteo
            # Mixing with Flattening (Kronecker Simplification)
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            x_mixed = x_opq
            # Flattening, (Bs, T*D)
            x_mixed = x_mixed.reshape(Bs, -1)
            # RMS normalization
            x_mixed = self.l_norm(x_mixed)
            # Reshaping and Kronecker Cross Mixing
            x_mixed = x_mixed.reshape(Bs, self.d_criteo, T*D//self.d_criteo)
            x_mixed = self.permutation_mixing(x_mixed).reshape(Bs, T, D)
            # Skip Connection
            out = x_opq + x_mixed

        elif self.mixing_type == 8882: # avazu
            # Mixing with Flattening (Kronecker Simplification)
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            x_mixed = x_opq
            # Flattening, (Bs, T*D)
            x_mixed = x_mixed.reshape(Bs, -1)
            # Reshaping and Kronecker Cross Mixing
            x_mixed = x_mixed.reshape(Bs, self.d_avazu, T*D//self.d_avazu)
            x_mixed = self.permutation_mixing(x_mixed).reshape(Bs, T, D)
            # Skip Connection
            out = self.l_norm(x_opq + x_mixed)

        elif self.mixing_type == 999:
            # Mixing with Flattening
            # Input: (Bs, T, D), Mixing: (Bs, T*D) -> (Bs, T*D), Output: (Bs, T, D)
            x_mixed = x_opq
            # Flattening, (Bs, T*D)
            x_mixed = x_mixed.reshape(Bs, -1)
            # RMS normalization
            x_mixed = self.pre_norm(x_mixed)
            # Cross Mixing and Reshaping
            x_mixed = self.cross_mixing(x_mixed).reshape(Bs, T, D)
            # Skip Connection
            out = x_opq + x_mixed

        if not is_batched:
            out = out.squeeze(0)

        return out


r'''
# ==========================================
# Test Code
# ==========================================

print("--- Test: Parametric OPQ with Eigenvalue Allocation ---")
T = 4
D = 16

# Init model
model = Token_Mixing(T, D, use_opq=True, opq_momentum=0.1, mixing_type=2).to('cuda')

# Create input with HIGHLY UNBALANCED variance
# Dim 0 has huge variance, Dim 1 small, etc.
# Ideally, allocation should distribute the "loud" dimensions across subspaces.
input_high_var = torch.randn(10000, T, D).to('cuda') * torch.linspace(1, 10, D).to('cuda')

print("Training with unbalanced data...")
model.train()
for i in range(5):
    _ = model(input_high_var)

# Check the first few columns of R
# If allocation worked, the columns of R should be a permuted set of Eigenvectors
print("Rotation Matrix (First 4x4 top-left):")
print(model.opq_R[:4, :4])

print("\nForward pass (Inference)...")
model.eval()
out = model(input_high_var)
print(f"Output shape: {out.shape}")
'''