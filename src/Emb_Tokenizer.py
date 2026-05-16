import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import RMSNorm

class Embedding_Tokenization(nn.Module):
    def __init__(self,
                 n_fields: int,
                 emb_size: int,
                 T: int,
                 D: int,
                 transform: bool = True,
                 mlp_params: dict = None):
        """
        Args:
            n_fields: Number of fields in the input.
            emb_size: Size of the embedding for each field.
            T: Number of parts (tokens) to divide the flattened input into.
            D: Output dimension for each token after MLP.
            mlp_params: Dictionary for MLP config (hidden_units, activation, dropout, use_layer_norm).
        """
        super(Embedding_Tokenization, self).__init__()

        self.n_fields = n_fields
        self.emb_size = emb_size
        self.T = T
        self.D = D
        self.transform = transform

        total_len = n_fields * emb_size

        # Judge whether T can divide n_fields
        check_fields = (n_fields % T == 0)

        # Print Judgements as requested
        print(
            f"n_field = {n_fields}, emb_size = {emb_size}, T = {T}, "
            f"token_fields = {n_fields // T if n_fields % T == 0 else round(n_fields / T, 2)}, "
            f"d = {total_len // T if total_len % T == 0 else round(total_len / T, 2)}"
        )

        if not check_fields:
            print(f"T ({T}) cannot divide n_fields ({n_fields}), but still applicable.")

        # Layer normalization
        self.pre_norm = RMSNorm(n_fields * emb_size)

        # Use projetion-based embedding tokenizer or not
        if self.transform is True:
            # Default MLP config if none provided
            defaults = {
                'hidden_units': [500],
                'activation': 'relu',
                'dropout': 0.1,
                'use_layer_norm': False,
                'trans_type': 'matrix'
            }
            if mlp_params is None: mlp_params = {}
            mlp_params = {**defaults, **mlp_params}

            # Build projection module
            self.trans_type = mlp_params.get('trans_type').lower()
            if self.trans_type == 'matrix':
                in_dim = int(total_len / T)
                out_dim = D
            elif self.trans_type == 'vector':
                in_dim = total_len
                out_dim = T*D
            else:
                raise ValueError("Currently only supporting Avazu and Criteo.")

            # Get activation function class
            act_fn_name = mlp_params.get('activation', 'relu').lower()
            if act_fn_name == 'relu':
                act_fn = nn.ReLU()
            elif act_fn_name == 'gelu':
                act_fn = nn.GELU()
            elif act_fn_name == 'silu':
                act_fn = nn.SiLU()
            elif act_fn_name == 'tanh':
                act_fn = nn.Tanh()
            elif act_fn_name == 'identity':
                act_fn = nn.Identity()
            else:
                act_fn = nn.ReLU()

            # Build hidden layers
            layers = []
            hidden_units = mlp_params.get('hidden_units', [])
            for h_dim in hidden_units:
                layers.append(nn.Linear(in_dim, h_dim))
                # post norm
                if mlp_params.get('use_layer_norm', False):
                    layers.append(RMSNorm(h_dim))
                layers.append(act_fn)
                if mlp_params.get('dropout', 0.0) > 0:
                    layers.append(nn.Dropout(mlp_params.get('dropout')))
                in_dim = h_dim

            # Final projection layer to map to D
            layers.append(nn.Linear(in_dim, out_dim))
            self.mlp = nn.Sequential(*layers)

        elif self.transform is False:
            print("Embedding tokenizer is NOT applied.")

        else:
            raise ValueError("Invalid transformation scheme.")

    def forward(self, x):
        """
        Input x shape: (Batch_Size, n_fields, emb_size) or (n_fields, emb_size)
        """
        # Handle batch dimension: if input is 2D, make it 3D for processing
        is_batched = True
        if x.dim() == 2:
            is_batched = False
            x = x.unsqueeze(0) # Shape: (1, n_fields, emb_size)

        Bs, _, _ = x.shape

        # Flattening: (Bs, n_fields, emb_size) -> (Bs, n_fields * emb_size)
        x_out = x.view(Bs, -1)
        # Layer normalization
        x_out = self.pre_norm(x_out)

        if self.transform is True:
            # When projection-based tokenizer is applied

            if self.trans_type == 'matrix':
                # Rankmixer-type tokenizer
                # Shape change: (Bs, total_len = n_fields * emb_size) -> (Bs, T, total_len/T)
                part_len = int((self.n_fields * self.emb_size) / self.T)
                x_out = x_out.reshape(Bs, self.T, part_len)

                # Shape change: (Bs, T, part_len) -> (Bs, T, D)
                # This effectively creates T vectors of size D and concatenates them along the T dim.
                x_out = self.mlp(x_out)

            elif self.trans_type == 'vector':
                # Full transformation on flattened embeddings
                # Shape change: (Bs, n_fields * emb_size) -> (Bs, T=Token_num * D=Token_dim)
                x_out = self.mlp(x_out)

                # (Bs, T=Token_num * D=Token_dim) -> (Bs, T=Token_num, D=Token_dim)
                x_out = x_out.reshape(Bs, self.T, self.D)

            else:
                raise ValueError("Invalid transformation scheme.")
            
        elif self.transform is False:
            # When NO tokenizer is applied, only shape change
            # (Bs, n_fields * emb_size) = (Bs, T * D) -> (Bs, T, D)
            x_out = x_out.reshape(Bs, self.T, self.D)

        # If input was not batched, remove the batch dimension to return (T, D)
        if not is_batched:
            x_out = x_out.squeeze(0)

        return x_out

r'''
# ==========================================
# Testing the Code
# ==========================================

# Configuration
n_fields = 10
emb_size = 32
T = 2  # 10 is divisible by 2, 10*32 (320) is divisible by 2
D = 64 # 64 is divisible by 2

mlp_config = {
    'hidden_units': [128, 64],
    'activation': 'relu',
    'dropout': 0.1,
    'use_layer_norm': True
}

print("--- Initializing Class ---")
try:
    model = Embedding_Tokenization(n_fields, emb_size, T, D, mlp_config).to('cuda')

    # Create Dummy Input (Batch Mode)
    # Shape: (Batch_Size=4, n_fields=10, emb_size=32)
    batch_input = torch.randn(4, n_fields, emb_size).to('cuda')

    print(f"\n--- Forward Pass (Batch Input: {batch_input.shape}) ---")
    output = model(batch_input)
    print(f"Output Shape: {output.shape}")
    # Expected: (4, T, D) -> (4, 2, 64)

    # Create Dummy Input (Single Mode)
    # Shape: (n_fields=10, emb_size=32)
    single_input = torch.randn(n_fields, emb_size).to('cuda')

    print(f"\n--- Forward Pass (Single Input: {single_input.shape}) ---")
    single_output = model(single_input)
    print(f"Output Shape: {single_output.shape}")
    # Expected: (T, D) -> (2, 64)

except ValueError as e:
    print(e)
'''