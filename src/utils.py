import numpy.random as random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

r'''
This .py file contains a set of built-in modules of neural networks & optimization methods.
'''

# --------------------------------------------------------------------------------------------------------------
# Global Response Normalization (Dropout / Normalizations /)
# --------------------------------------------------------------------------------------------------------------

class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer from the paper 
    "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (CVPR 2023)
    Typically LN -> Conv (Nonlinearity) -> GRN

    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# --------------------------------------------------------------------------------------------------------------
# RMS Normalization (Dropout / Normalizations /)
# --------------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x**2).mean(-1, keepdim=True)
        out_mean = x / torch.sqrt(mean + self.eps) # root mean square
        out = self.gamma * out_mean 
        return out

# --------------------------------------------------------------------------------------------------------------
# SwiGLU-like FFN (Neural Networks)
# --------------------------------------------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU-like FFN from paper "GLU Variants Improve Transformer (arXiv 2020)" 
    SwiGLU(x) = Activation(Linear1(x)) * (Linear2(x))
    Args:
        emb_dim: embedding dimension, supposing input be (..., emb_dim)
        expansion_rate: expansion of representation space
    """
    def __init__(self, emb_dim:int, expansion_rate:float=2.0, bias:bool=True):
        super().__init__()
        self.exp_rate = expansion_rate
        self.fc_lift1 = nn.Linear(emb_dim, int(emb_dim * expansion_rate), bias=bias)
        self.fc_lift2 = nn.Linear(emb_dim, int(emb_dim * expansion_rate), bias=bias)
        self.skip1 = nn.Linear(emb_dim, emb_dim, bias=bias)
        if self.exp_rate != 1.0:
            self.fc_reduce = nn.Linear(int(emb_dim * expansion_rate), emb_dim, bias=bias)
            self.skip2 = nn.Linear(emb_dim, int(emb_dim * expansion_rate), bias=bias)
        # self.activation = nn.SiLU()
        # self.activation = nn.ReLU()
        self.activation = nn.GELU()

    def forward(self, x):
        x_fc1 = self.fc_lift1(x)
        x_fc2 = self.fc_lift2(x)
        if self.exp_rate != 1.0:
            x = self.fc_reduce(
                (self.skip2(x) + self.activation(x_fc1)) * x_fc2) + self.skip1(x)
        else:
            x = (self.skip1(x) + self.activation(x_fc1)) * x_fc2 + self.skip1(x)
        return x


# --------------------------------------------------------------------------------------------------------------
# SwiGLU-like FFN + mHC (Neural Networks)
# --------------------------------------------------------------------------------------------------------------

class SwiGLU_MHC(nn.Module):
    """
    SwiGLU-like FFN + mHC (Manifold-Constrained Hyper-Connections) 
    from paper "mHC: Manifold-Constrained Hyper-Connections (arXiv 2025)" 
    Args:
        emb_dim: embedding dimension, supposing input be (..., emb_dim)
        expansion_rate: expansion of representation space
        rate: head number of mHC 
        max_sk_it: maximum iteration of sinkhorn knopp algorithm
        output_layer: output summation (or not)
    """
    def __init__(self, emb_dim:int, expansion_rate:float=2.0, rate:int=4, 
                 bias:bool=True, max_sk_it:int=20):
        super().__init__()
        self.emb_dim = emb_dim
        self.expansion_rate = expansion_rate
        self.rate = rate
        self.bias = bias
        self.max_sk_it = max_sk_it

        # Layer normalization
        self.ffn_norm = RMSNorm(emb_dim)

        # mHC (Manifold-constrained Hyperconnection)
        self.ffn_mHC = HyperConnection(
            dim=emb_dim,
            rate=rate,
            max_sk_it=max_sk_it,
            bias=bias
        )

        # SwiGLU type FFN
        self.ffn = SwiGLU(
            emb_dim=emb_dim,
            expansion_rate=expansion_rate,
            bias=bias
        )

    def forward(self, h):
        # SwiGLU-type FFN with Manifold-constrained Hyperconnection
        # Input: (Bs, D) or (Bs, T, D) for the first layer, (Bs, 1 or T, [rate], D) for others
        # Output: (Bs, 1 or T, [rate], D) for non-last layer, (Bs, D) or (Bs, T, D) for the last layer

        # (Bs, D) or (Bs, T, D) -> (Bs, T, [rate], D)
        if h.dim() == 2: 
            # (Bs, D) as input
            h = h.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.rate, 1)
        elif h.dim() == 3:
            # (Bs, T, D) as input
            h = h.unsqueeze(2).repeat(1, 1, self.rate, 1)

        # Mapping
        # (Bs, T, [rate], D) -> (Bs, T, [rate], D)
        mix_h_ffn, beta_ffn = self.ffn_mHC.width_connection(h)
        h_in_ffn = self.ffn_norm(mix_h_ffn[..., 0, :])
        ffn_output = self.ffn(h_in_ffn)
        h = self.ffn_mHC.depth_connection(mix_h_ffn, ffn_output, beta_ffn)

        return h
        
class HyperConnection(nn.Module):
    """
    Manifold-constrainted Hyper-Connection module replaces the traditional residual connection with
    width connection and depth connection. The dynamic mode optionally generates connection weights.

    mHC upgrades:
    1. Enforce non-negativity on H_pre and H_post via Sigmoid.
    2. Project H_res onto the doubly stochastic manifold using Sinkhorn-Knopp.
    3. Use Flattening (not Mean) for dynamic input mapping.
    4. Use Mean (not Sum) for the final aggregation.
    """
    def __init__(self, dim:int, rate:int, dynamic:bool=True, 
                 max_sk_it:int=20, bias:bool=True):
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.dynamic = dynamic
        self.max_sk_it = max_sk_it
        
        # --- Static mapping initialization ---
        # Recommended static mappings should be initialized close to
        # an identity mapping or a uniform distribution.
        # For H_res: the identity matrix serves as the identity-mapping initializer.
        init_H_res = torch.eye(rate)
        # For H_pre: [1, 0, 0, ..., 0] preserves the 0-th flow as the layer input.
        init_H_pre = torch.zeros(rate, 1)
        init_H_pre[0, 0] = 1.0
        # Concatenate to static_alpha: [rate, rate + 1]
        self.static_alpha = nn.Parameter(torch.cat([init_H_pre, init_H_res], dim=1))
        # For H_post (beta): initialize with a uniform distribution 
        # [1/rate, 1/rate, ..., 1/rate].
        self.static_beta = nn.Parameter(torch.ones(rate) / rate)

        if self.dynamic:
            self.layer_norm = RMSNorm(dim*rate)
            # --- Key modification 1: use Flattening for dynamic mapping input (Eq. 7) ---
            # The input dimension should be rate * dim (flattened) to retain full context.
            self.dynamic_alpha_fn = nn.Linear(rate * dim, rate * (rate + 1), bias=bias)
            self.dynamic_beta_fn = nn.Linear(rate * dim, rate, bias=bias)
            
            # Appendix A.1 notes that the gating factor should be initialized to a small value (0.01).
            self.dynamic_alpha_scale = nn.Parameter(torch.full((1,), 0.01))
            self.dynamic_beta_scale = nn.Parameter(torch.full((1,), 0.01))

    def width_connection(self, h: torch.Tensor):
        B, L, N, D = h.shape # [B, L, rate, dim]
        
        if self.dynamic:
            # --- Key modification 2: apply Flattening then RMS_Norm ---
            agg_h = h.view(B, L, -1) # [B, L, rate*dim]
            agg_h = self.layer_norm(agg_h)
            
            dyn_alpha = torch.tanh(self.dynamic_alpha_fn(agg_h)).view(B, L, self.rate, self.rate + 1)
            dyn_beta = self.dynamic_beta_fn(agg_h).view(B, L, self.rate)
            
            # Apply the gating scale and add it to the static component
            alpha = self.static_alpha + dyn_alpha * self.dynamic_alpha_scale
            beta = self.static_beta + dyn_beta * self.dynamic_beta_scale
        else:
            alpha = self.static_alpha.expand(B, L, -1, -1)
            beta = self.static_beta.expand(B, L, -1)

        # mHC core constraints
        # Split alpha: [B, L, rate, rate+1] -> H_pre_raw [B, L, rate, 1], 
        # H_res_raw: [B, L, rate, rate]
        H_pre_raw = alpha[..., :1]      # [B, L, rate, 1]
        H_res_raw = alpha[..., 1:]      # [B, L, rate, rate]
        
        # --- Key modification 3: non-negativity constraint on H_pre (Eq. 8) ---
        H_pre = torch.sigmoid(H_pre_raw)
        
        # --- Key modification 4: Sinkhorn projection of H_res onto the doubly stochastic manifold ---
        H_res = sinkhorn_knopp(H_res_raw, self.max_sk_it, 1e-12)
        
        # --- Key modification 5: non-negativity constraint on H_post (beta) (Eq. 8) ---
        # Equation (8): H_post = 2 * sigmoid(~H_post)
        beta_constrained = 2.0 * torch.sigmoid(beta)

        # Applying both H_pre and H_res
        # h: [B, L, rate, dim]
        # h_for_layer: H_pre^T @ h -> [B, L, 1, dim]
        h_for_layer = torch.matmul(H_pre.transpose(-2, -1), h)
        # h_res_flow: H_res^T @ h -> [B, L, rate, dim]
        h_res_flow = torch.matmul(H_res.transpose(-2, -1), h)
        
        # Concatenation on residual flows mix_h[..., 0, :] 
        # Serving as the input to Layer F.
        mix_h = torch.cat([h_for_layer, h_res_flow], dim=-2) # [B, L, rate+1, dim]
        
        return mix_h, beta_constrained

    def depth_connection(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # mix_h: [B, L, rate+1, dim]
        h_prime = mix_h[..., 1:, :] # Residual flow
        # h_o: output of Layer F with shape [B, L, dim]
        # Corresponding to Eq.3: x^{l+1} = H_res^l x^l + (H_post^l)^T F(...)
        # h_o_weighted: (H_post^l)^T F(...) -> [B, L, rate, dim]
        h_o_weighted = torch.einsum('bld,bln->blnd', h_o, beta)
        return h_prime + h_o_weighted

def sinkhorn_knopp(matrix: torch.Tensor, num_iter: int = 20, epsilon: float = 1e-20) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm: projects a matrix onto the doubly stochastic manifold.
    A doubly stochastic matrix has: all row sums = 1, all column sums = 1, 
    and all entries ≥ 0.
    Args:
        matrix: input matrix of shape [batch_size, n, n]
        num_iter: number of iterations; mHC paper recommends 20
        epsilon: small constant for numerical stability to avoid division by zero
    Returns:
        A doubly stochastic matrix with the same shape as the input
    """
    # Ensure the input matrix is non-negative (typically after exponentiation)
    # Subtract the max value to avoid exp overflow (a variant of the log-sum-exp trick)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = matrix - torch.max(matrix, dim=-1, keepdim=True)[0]
    K = torch.exp(matrix)
    for _ in range(num_iter):
        # Row normalization - sum of row elements being 1
        K = K / (K.sum(dim=-1, keepdim=True) + epsilon)
        # Col normalization - sum of col elements being 1
        K = K / (K.sum(dim=-2, keepdim=True) + epsilon)
    return K

class MeanOn4(nn.Module):
    def forward(self, x):
        # If shape is (B, T, rate, D)
        if x.dim() == 4:
            x = x.mean(dim=2)  # (B, T, D)
            if x.size(1) == 1:
                return x.squeeze(1)  # (B, D)
            return x  # (B, T, D)
        # Otherwise pass
        return x

# --------------------------------------------------------------------------------------------------------------
# DynaMixer Block (Neural Networks)
# --------------------------------------------------------------------------------------------------------------


class DynaMixerBlock(nn.Module):
    '''
    DynaMixer from the paper "DynaMixer: A Vision MLP Architecture with Dynamic Mixing" (ICML 2022)
    Args:
        dim: feature dimension, supposing (Bs, H, W, dim=C) as shape of input.
        resolution: equal to H.
        num_head: dim // num_head >= 1
    Output:
        x: shape as (Bs, H, W, dim=C), invariant in shape.
    '''
    def __init__(self, dim, resolution=32, num_head=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.resolution = resolution
        self.num_head = num_head
        self.mix_h = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mix_w = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = ReweightNN(dim, dim * 3, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        res_connect = x

        h = self.mix_h(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        w = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x) + res_connect

        return x

class ReweightNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DynaMixerOp(nn.Module):
    def __init__(self, dim, seq_len, num_head, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        BW, H, C = x.shape
        # weights: (BW, num_head, H * reduced_dim)
        weights = self.compress(x).reshape(BW, H, self.num_head, self.reduced_dim)
        weights = weights.permute(0, 2, 1, 3).reshape(BW, self.num_head, -1)
        # weights: (BW, num_head, H, H)
        weights = self.generate(weights).reshape(BW, self.num_head, H, H)
        weights = self.activation(weights)
        # Mixing
        res_connect = x
        x = x.reshape(BW, H, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(BW, H, C)
        x = self.out(x) + res_connect
        return x


# --------------------------------------------------------------------------------------------------------------
# SwiGLU-CrossNet (Neural Networks)
# --------------------------------------------------------------------------------------------------------------

class SwiCrossV2(nn.Module):
    """
    A CrossNetV2 module with SwiGLU nonlinearity. 
    SwiGLU(x) = SiLU(Linear1(x)) * (Linear2(x))
    Args:
        input_dim: flattened embedding dimension, supposing input be (Bs, input_dim)
        num_layers: total order of feature interactions
    """
    def __init__(self, input_dim, num_layers, bias:bool=True, drop:float=0.1):
        super(SwiCrossV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim, bias=bias)
                                          for _ in range(self.num_layers))
        self.id_layers = nn.ModuleList(nn.Linear(input_dim, input_dim, bias=bias)
                                          for _ in range(self.num_layers))
        self.act = nn.ReLU()
        # self.drop = nn.Dropout(drop)

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.act(self.id_layers[i](X_i) * self.cross_layers[i](X_i))
        return X_i
    

# --------------------------------------------------------------------------------------------------------------
# Low-rank Flattening Mixing (Neural Networks)
# --------------------------------------------------------------------------------------------------------------

class LRFlaMix(nn.Module):
    """ Low-rank Flattening Mixing Module motivated by CrossNetMix:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    """
    def __init__(self, in_features, low_rank=32, num_experts=4):
        super(LRFlaMix, self).__init__()
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank)))
        # V: (in_features, low_rank)
        self.V = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank)))
        # C: (low_rank, low_rank)
        self.C = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank)))
        # gating list
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1)))
        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        output_of_experts = []
        gating_score_of_experts = []
        for expert_id in range(self.num_experts):
            # (1) G(x_0)
            # compute the gating score by x_0
            gating_score_of_experts.append(self.gating[expert_id](x_0.squeeze(2)))

            # (2) E(x_0)
            # project the input x_0 to $\mathbb{R}^{r}$
            v_x = torch.matmul(self.V[expert_id].t(), x_0)  # (bs, low_rank, 1)

            # nonlinear activation in low rank space
            v_x = torch.tanh(v_x)
            v_x = torch.matmul(self.C[expert_id], v_x)
            v_x = torch.tanh(v_x)

            # project back to $\mathbb{R}^{d}$
            uv_x = torch.matmul(self.U[expert_id], v_x)  # (bs, in_features, 1)

            dot_ = uv_x + self.bias

            output_of_experts.append(dot_.squeeze(2))

        # (3) mixture of low-rank experts
        output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
        gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
        moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
        x_0 = moe_out + x_0  # (bs, in_features, 1)

        x_0 = x_0.squeeze()  # (bs, in_features)
        return x_0