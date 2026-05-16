import torch
import torch.nn as nn
from .Token_FFN import Token_Interaction
from .Token_Mixing import Token_Mixing
import torch.nn.functional as F
from .utils import RMSNorm

class UniMixer_layer(nn.Module):
    def __init__(self,
                 T: int,
                 D: int,
                 # --- Token Mixing Config ---
                 use_opq: bool = False,
                 opq_momentum: float = 0.01,
                 mixing_type: int = 2,
                 mixing_mlp_params: dict = None,
                 mixing_attn_params: dict = None,
                 # --- Token Interaction Config ---
                 interaction_ffn_type: str = 'mlp',
                 interaction_mlp_params: dict = None):
        """
        Args:
            T: Number of tokens (and number of subspaces H).
            D: Embedding dimension.

            [Token Mixing Params]
            use_opq: Enable Parametric OPQ (Eigenvalue Allocation).
            opq_momentum: Momentum for online covariance updates.
            mixing_type: 1(Identity), 2(Permute), 3(Linear), 4(MLP), 5(Attention).
            mixing_mlp_params: Config for Type 4.
            mixing_attn_params: Config for Type 5.

            [Token Interaction Params]
            interaction_ffn_type: 'mlp' (default) or 'other'.
            interaction_mlp_params: Config for the independent per-token FFNs.
        """
        super(UniMixer_layer, self).__init__()

        # Token Mixing Module
        # Responsible for: OPQ (Normalization) -> Mixing (Features/Tokens) -> Residual -> Norm
        self.token_mixing = Token_Mixing(
            T=T,
            D=D,
            use_opq=use_opq,
            opq_momentum=opq_momentum,
            mixing_type=mixing_type,
            mlp_params=mixing_mlp_params,
            attention_params=mixing_attn_params
        )

        # Token Interaction Module
        # Responsible for: Independent FFN per token -> Residual -> Norm
        # Note: We pass H=T, because the output of Mixing is (Bs, T, D)
        self.token_interaction = Token_Interaction(
            H=T, # H equals T in this architecture
            T=T,
            D=D,
            ffn_type=interaction_ffn_type,
            mlp_params=interaction_mlp_params
        )

        # Normalization
        self.layer_norm1 = RMSNorm(D)
        # self.layer_norm2 = RMSNorm(D)

    def forward(self, x):
        """
        Input x: (Bs, T, D) - typically from Embedding_Tokenization or previous UniMixer_layer
        Output:  (Bs, T, D)
        """
        # Global / Cross Mixing (w. / w.o. OPQ)
        # Shape: (Bs, T, D) -> (Bs, T, D)
        x_mixed = self.token_mixing(x)

        # Layer normalization
        x_mixed = self.layer_norm1(x_mixed)

        # Per-Token Independent Interaction
        # Shape: (Bs, T, D) -> (Bs, T, D)
        x_out = self.token_interaction(x_mixed)
        
        # Layer normalization
        # x_out = self.layer_norm2(x_out)

        return x_out


r'''
# ==========================================
# Test Code
# ==========================================

print("--- Test: UniMixer layer ---")


T = 4
D = 16

# Configure the layer
layer = UniMixer_layer(
    T=T,
    D=D,
    # Mixing Config (e.g., using Attention)
    use_opq=True,
    opq_momentum=0.01,
    mixing_type=4,
    mixing_attn_params={'dropout': 0.1},
    # Interaction Config (e.g., using default MLPs)
    interaction_ffn_type='mlp',
    interaction_mlp_params={'hidden_units': [64], 'dropout': 0.1}
).to('cuda')

# Dummy Batch Input
bs = 2
input_tensor = torch.randn(bs, T, D).to('cuda')

print(f"Input Shape: {input_tensor.shape}")

# Forward Pass
# If using OPQ, remember it updates stats during .train()
layer.train()
output_tensor = layer(input_tensor)

print(f"Output Shape: {output_tensor.shape}")
'''