import torch
import torch.nn as nn
from .utils import DynaMixerBlock, SwiCrossV2, SwiGLU, SwiGLU_MHC
from .utils import GRN, MeanOn4, RMSNorm
from fuxictr.pytorch.layers import CrossNetV2
import torch.nn.functional as F

class Token_Interaction(nn.Module):
    def __init__(self,
                 H: int,
                 T: int,
                 D: int,
                 ffn_type: str = 'mlp',
                 mlp_params: dict = None):
        """
        Args:
            H: Number of subspaces (rows).
            T: Number of tokens (used to calc feature length).
            D: Embedding size (used to calc feature length).
            ffn_type: 'mlp' (default) or 'other' (TODO).
            mlp_params: Configuration for the MLPs (hidden_units, activation, dropout).
        """
        super(Token_Interaction, self).__init__()

        self.H = H
        self.T = T
        self.D = D
        self.ffn_type = ffn_type.lower()

        # Calculate Input/Output Feature Dimension for each FFN
        # Input shape per row: T * (D / H)
        # Note: In previous context, H=T, so feature_dim = T * (D/T) = D.
        # if D % H != 0:
        #     raise ValueError(f"D ({D}) must be divisible by H ({H})")

        self.feature_dim = T * D // H

        print(f"Token_Interaction Init: H={H}, Feature Dim per FFN={self.feature_dim}")

        # Build H Parallel FFNs
        # We use ModuleList to hold independent networks
        self.parallel_ffns = nn.ModuleList()

        if self.ffn_type == 'mlp':
            # Per-token FFNs, built with H Parallel FFNs
            # We use ModuleList to hold independent networks
            self.parallel_ffns = nn.ModuleList()

            # Setup Defaults
            defaults = {
                'hidden_units': [64] * 3, # Standard Setting
                'activation': 'swiglu',
                'dropout': 0.1,
                'expansion_rate': 2.0,
                'bias': True,
                'use_layer_norm': False,
            }
            # Merge defaults
            if mlp_params is None: mlp_params = {}
            params = {**defaults, **mlp_params}

            # Create H independent MLPs
            for _ in range(H):
                self.parallel_ffns.append(
                    self._build_mlp(self.feature_dim, self.feature_dim, params)
                )

        # elif self.ffn_type == 'widemlp':
        #     # Unified MLP for replacing per-token FFNs
        #     # Both non-linearity and wide hidden space are necessary

        #     # Pre-Norm
        #     # self.pre_norm = nn.LayerNorm(self.feature_dim)

        #     # Setup Defaults
        #     if mlp_params is None: mlp_params = {}
        #     defaults = {
        #         'hidden_units': [400] * 3, # Standard Setting
        #         'activation': 'swiglu',
        #         'dropout': 0.1
        #     }
        #     # Merge defaults
        #     params = {**defaults, **mlp_params}

        #     # Create an unified MLP
        #     self.ffn = self._build_mlp(self.feature_dim, self.feature_dim, params)
            
        #     # Post-Norm
        #     # self.layer_norm = nn.LayerNorm(self.feature_dim)

        # elif self.ffn_type == 'dynamixer':
        #     # Setup Defaults
        #     # Please be aware "dim" and "resolution" here is consistent with (D,H,T)
        #     # Only the rest parameters can be changed here!

        #     # Pre-Norm
        #     self.pre_norm = nn.LayerNorm(self.feature_dim)

        #     ''' (1) Single-Layer FFN
        #     '''
        #     self.ffn = DynaMixerBlock(dim=self.D//self.H, resolution=self.H, num_head=self.D//self.H)

        #     ''' (2) Multi-Layer FFN
        #     '''
        #     # self.ffn = nn.Sequential(
        #     #     GRN(self.feature_dim // self.T),
        #     #     DynaMixerBlock(dim=self.D//self.H, resolution=self.H, num_head=self.D//self.H),
        #     #     GRN(self.feature_dim // self.T),
        #     #     DynaMixerBlock(dim=self.D//self.H, resolution=self.H, num_head=self.D//self.H)
        #     # )

        #     # Post-Norm
        #     self.layer_norm = GRN(self.feature_dim // self.T)

        # elif self.ffn_type == 'dcnv2':
        #     # Parallel DCNv2, built with [num_expert] DCNv2 Interactions
        #     # We use ModuleList to hold independent networks
        #     self.parallel_ffns = nn.ModuleList()

        #     # Pre-Norm
        #     self.pre_norm = nn.LayerNorm(self.feature_dim)

        #     # Create [num_expert] Parallelized DCNv2
        #     input_dim = self.H * self.feature_dim
        #     num_layer = 3
        #     num_expert = 5
        #     for _ in range(num_expert):
        #         self.parallel_ffns.append(
        #             SwiCrossV2(input_dim, num_layer)
        #         )

        #     # Concatenation then Linear
        #     self.linear_cat = nn.Linear(num_expert * self.feature_dim, self.feature_dim)

        #     # Post-Norm
        #     self.layer_norm = nn.LayerNorm(self.feature_dim)

        # elif self.ffn_type == 'dcnffn_cat':
        #     # Interaction with Linear([DNN(X),DCNv2(X)])

        #     # Pre-Norm
        #     self.pre_norm = nn.LayerNorm(self.feature_dim)

        #     # Setup Defaults
        #     if mlp_params is None: mlp_params = {}
        #     defaults = {
        #         'hidden_units': [500] * 3, # Standard Setting
        #         'activation': 'relu',
        #         'dropout': 0.1
        #     }
        #     # Merge defaults
        #     params = {**defaults, **mlp_params}

        #     # Create H independent MLPs
        #     # for _ in range(H):
        #     #     self.parallel_ffns.append(
        #     #         self._build_mlp(self.feature_dim, self.feature_dim, params)
        #     #     )
        #     self.parallel_ffns = self._build_mlp(self.H * self.feature_dim, self.H * self.feature_dim, params)
            
        #     # Create CrossNet interactions
        #     self.crossnet = CrossNetV2(self.H * self.feature_dim, 3)

        #     # Concatenation then Linear
        #     self.linear_cat = nn.Linear(2 * self.H * self.feature_dim, self.H * self.feature_dim)
            
        #     # Post-Norm
        #     self.layer_norm = nn.LayerNorm(self.feature_dim)

        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def _build_mlp(self, input_dim, output_dim, params):
        """Helper to build a single MLP"""
        layers = []
        in_d = input_dim

        hidden_units = params['hidden_units']
        act_name = params['activation'].lower()
        dropout = params['dropout']
        layer_norm = params['use_layer_norm']

        # Select Activation
        if act_name == 'relu': act_fn = nn.ReLU()
        elif act_name == 'gelu': act_fn = nn.GELU()
        elif act_name == 'tanh': act_fn = nn.Tanh()
        elif act_name == 'swiglu': act_fn = SwiGLU
        elif act_name == 'swiglu-mhc': act_fn = SwiGLU_MHC
        else: act_fn = nn.GELU()

        if act_name in ['swiglu', 'swiglu-mhc']:
            # SwiGLU-FFN / SwiGLU-FFN with mHC
            for h_dim in hidden_units:
                # Pre-norm
                if layer_norm:
                    layers.append(RMSNorm(in_d))
                layers.append(act_fn(
                    emb_dim=in_d,
                    expansion_rate=params['expansion_rate'],
                    bias=params['bias']))
                # Post-norm
                # if layer_norm:
                #     layers.append(RMSNorm(in_d))
                if h_dim != in_d:
                    layers.append(nn.Linear(
                        in_features=in_d, 
                        out_features=h_dim, 
                        bias=params['bias']))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_d = h_dim

            layers.append(MeanOn4())
        
        else:
            for h_dim in hidden_units:
                layers.append(nn.Linear(
                    in_features=in_d, 
                    out_features=h_dim,
                    bias=params['bias']))
                if layer_norm:
                    layers.append(RMSNorm(h_dim))
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_d = h_dim

        # Projection back to original size
        layers.append(nn.Linear(in_d, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input x: (Bs, H, T*D/H) or (H, T*D/H)
        """
        # Handle Batch Logic
        is_batched = True
        if x.dim() == 2:
            is_batched = False
            x = x.unsqueeze(0) # (1, H, Feat_Dim=T*D//H)

        Bs, H, Feat_Dim = x.shape

        if H != self.H:
            raise ValueError(f"Input H ({H}) does not match initialized H ({self.H})")
        if Feat_Dim != self.feature_dim:
            raise ValueError(f"Input Feature Dim ({Feat_Dim}) does not match expected ({self.feature_dim})")


        if self.ffn_type == 'mlp':
            # Parallel Processing with H independent FFNs
            ffn_outputs = []

            # Iterate over the H dimension
            for h in range(self.H):
                # Extract the h-th subspace/row for the whole batch
                # Shape: (Bs, Feature_Dim)
                x_h = x[:, h, :]

                # Apply the h-th independent FFN
                # Shape: (Bs, Feature_Dim)
                out_h = self.parallel_ffns[h](x_h)

                ffn_outputs.append(out_h)

            # Stack them back together
            # List of (Bs, D) -> (Bs, H, D)
            x_ffn = torch.stack(ffn_outputs, dim=1)

            # Residual Connection + Layer Norm
            # Add original input x
            out = x_ffn + x


        # elif self.ffn_type == 'widemlp':
        #     # Unified MLP for replacing per-token FFNs
        #     # Both non-linearity and wide hidden space are necessary
        #     res_connect = x
        #     # x = self.pre_norm(x)

        #     # Unified FFN interaction
        #     x_ffn = self.ffn(x)

        #     # Residual Connection + Layer Norm
        #     # Add original input x
        #     # out = self.layer_norm(x_ffn) + res_connect # Peri-Norm
        #     # out = self.layer_norm(x_ffn + res_connect) # Post-Norm
        #     out = x_ffn + res_connect # No Norm       


        # elif self.ffn_type == 'dynamixer':
        #     # Dynamixer for feature interaction
        #     # Pre-Norm
        #     x_norm = self.pre_norm(x)
        #     # (Bs, H, Feat_Dim=T*D//H) -> (Bs, H, T, D//H)
        #     x_norm = x_norm.reshape(Bs, H, self.T, Feat_Dim//self.T)
        #     # Dynamixer Interaction
        #     # (Bs, H, T, D//H) -> (Bs, H, T*D//H)
        #     x_ffn = self.ffn(x_norm)

        #     # Residual Connection + Layer Norm
        #     # Add original input x
        #     out = x + self.layer_norm(x_ffn).reshape(Bs, H, -1)


        # elif self.ffn_type == 'dcnv2':
        #     # Parallelized DCNv2 Interactions
        #     # Using [num_expert] parallel Interaction Functions

        #     # Pre-Norm
        #     res_connect = x
        #     # x = self.pre_norm(x)

        #     ffn_outputs = []

        #     # Flattening; (Bs, H, T*D/H) -> (Bs, T*D)
        #     x = x.reshape(x.size(0), -1)

        #     # Iterate over all DCNv2 experts
        #     for expert in self.parallel_ffns:
        #         # Reshape output back to (Bs, H, T*D/H)
        #         ffn_outputs.append(expert(x).reshape(Bs, H, Feat_Dim))

        #     # Concatenation then Linear
        #     x_ffn = torch.cat(ffn_outputs, dim=-1)
        #     out = self.linear_cat(x_ffn) + res_connect

        #     # Post-Norm
        #     # out = self.layer_norm(out)


        # elif self.ffn_type == 'dcnffn_cat':
        #     # Parallelized DCNv2 and Per-token FFNs
        #     # Concatenation then Linear

        #     # Pre-Norm
        #     # x = self.pre_norm(x)

        #     '''Per-token FFN Part
        #     '''
        #     # ffn_outputs = []

        #     # # Iterate over the H dimension
        #     # for h in range(self.H):
        #     #     # Extract the h-th subspace/row for the whole batch
        #     #     # Shape: (Bs, Feature_Dim)
        #     #     x_h = x[:, h, :]

        #     #     # Apply the h-th independent FFN
        #     #     # Shape: (Bs, Feature_Dim)
        #     #     out_h = self.parallel_ffns[h](x_h)

        #     #     ffn_outputs.append(out_h)

        #     # # Stack them back together
        #     # # List of (Bs, D) -> (Bs, H, D)
        #     # x_ffn = torch.stack(ffn_outputs, dim=1)

        #     x_ffn = self.parallel_ffns(x.reshape(x.size(0), -1))

        #     '''DCNv2 Part
        #     '''
        #     # Flattening; (Bs, H, T*D/H) -> (Bs, T*D)
        #     x_dcn = x.reshape(x.size(0), -1)

        #     # DCNv2 interaction
        #     x_dcn = self.crossnet(x_dcn)

        #     # Residual Connection
        #     # Add original input x
        #     out = self.linear_cat(torch.cat([x_ffn,x_dcn], dim=-1)) 
        #     out = x + out.reshape(Bs, H, Feat_Dim) # No Norm

        #     # Post-Norm
        #     # out = self.layer_norm(out)

        # Restore shape for single input case
        if not is_batched:
            out = out.squeeze(0)

        return out

r'''
# ==========================================
# Testing the Code
# ==========================================

print("--- Initializing Token_Interaction ---")

# Config matches previous context (H=T)
H = 4
T = 4
D = 16

# 1. Instantiate
# Feature dim will be T * (16/4) = 4 * 4 = 16
model = Token_Interaction(H, T, D, ffn_type='mlp', mlp_params={'hidden_units': [32]})

print(f"Number of Independent FFNs: {len(model.parallel_ffns)}")

# 2. Test Batch Input
# Input Shape: (Bs=2, H=4, Feature_Dim=16)
batch_input = torch.randn(2000, H, D)
output = model(batch_input)

print(f"\nBatch Input Shape: {batch_input.shape}")
print(f"Output Shape:      {output.shape}")

# 3. Test GPU Compatibility
if torch.cuda.is_available():
    print("\n--- Testing GPU Support ---")
    device = torch.device('cuda')
    model = model.to(device)
    gpu_input = batch_input.to(device)
    gpu_output = model(gpu_input)
    print(f"Output Device: {gpu_output.device}")
else:
    print("\nSkipping GPU test (CUDA not available)")
'''