# import torch
import torch.nn as nn
from .UniMixer_layer import UniMixer_layer
from .Emb_Tokenizer import Embedding_Tokenization
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
import torch.nn.functional as F

class UniMixer(BaseModel):
    def __init__(self, 
                 feature_map,
                 Token_num: int=10,
                 Token_dim: int=10,
                 model_id="UniMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(UniMixer, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        
        self.T = kwargs.get("Token_num", Token_num)
        self.D = kwargs.get("Token_dim", Token_dim)
        self.depth = kwargs.get("depth", 2)
        self.embed_size = kwargs.get("embedding_dim", embedding_dim)
        self.use_tokenizer = kwargs.get("use_tokenizer", True)

        input_dim = feature_map.sum_emb_out_dim()
        n_fields = input_dim // self.embed_size
        
        # Initial Embedding
        self.embedding_layer = FeatureEmbedding(feature_map, self.embed_size)

        # Embedding & Tokenization
        # Input: (Bs, n_fields, emb_size) -> Output: (Bs, T, D)
        if self.use_tokenizer: # Use tokenizer
            self.embedding_tokenization = Embedding_Tokenization(
                n_fields=n_fields,
                emb_size=self.embed_size,
                T=self.T,
                D=self.D,
                transform=True,
                mlp_params=kwargs.get("proj_param", None)
            )
        else: # No tokenizer
            check_equality = (n_fields * self.embed_size == self.T * self.D)

            if not check_equality:
                raise ValueError(f"Inconsistent shape.")
            
            self.embedding_tokenization = Embedding_Tokenization(
                n_fields=n_fields,
                emb_size=self.embed_size,
                T=self.T,
                D=self.D,
                transform=False,
                mlp_params=kwargs.get("proj_param", None)
            )
            

        # Backbone (L layers of UniMixer)
        # Input: (Bs, T, D) -> Output: (Bs, T, D)
        # We support hybrid mixing here
        mixing_type = kwargs.get("mixing_type")
        if type(mixing_type) == int:
            self.mixing_type = [mixing_type] * self.depth
        elif type(mixing_type) == list:
            self.mixing_type = mixing_type

        # Layer stacking
        self.backbone = nn.ModuleList()
        for i in range(self.depth):
            # OPQ for deeper layers can be disabled if required
            # by adding logic here (e.g., use_opq = (i == 0))
            # We use it for all layers here based on "Dynamic Normalization" logic.
            self.backbone.append(
                UniMixer_layer(
                    T=self.T,
                    D=self.D,
                    # --- Token Mixing Config ---
                    use_opq=kwargs.get("use_opq", False),
                    opq_momentum=kwargs.get("opq_momentum", 0.01),
                    mixing_type=self.mixing_type[i],
                    mixing_mlp_params=kwargs.get("mixing_mlp_params", None),
                    mixing_attn_params=kwargs.get("mixing_attn_params", None),
                    # --- Token Interaction Config ---
                    interaction_ffn_type=kwargs.get("interaction_ffn_type", 'mlp'),
                    interaction_mlp_params=kwargs.get("interaction_mlp_params", None)
                )
            )

        # Prediction Head Logic
        self.head_type = kwargs.get("pred_head", 'noparam_mean')
        if self.head_type == 'flatten':
            # Trainable: Flatten + Linear
            self.pred = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.T * self.D, 1)
            )
        elif self.head_type == 'mlp':
            # Trainable: Flatten + MLP
            self.pred = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.T * self.D, self.T * self.D * 2),
                nn.ReLU(),
                nn.Linear(self.T * self.D * 2, 1)
            )
        elif self.head_type == 'noparam_mean':
            # Non-Parametric: Pure averaging. No weights.
            pass
        else:
            raise ValueError("head_type must be 'flatten' or 'noparam_mean'")

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)

        # Transform raw data into embeddings
        # x -> (Bs, n_fields, emb_size)
        feature_emb = self.embedding_layer(X)

        # Tokenize embeddings
        # (Bs, n_fields, emb_size) -> (Bs, T, D)
        tokens = self.embedding_tokenization(feature_emb)

        # Token mixing
        # (Bs, T, D) -> (Bs, T, D)
        for layer in self.backbone:
            tokens = layer(tokens)

        # Prediction
        # (Bs, T, D) -> (Bs, 1)
        if self.head_type == 'flatten':
            # Flatten -> (Bs, T*D) -> Linear -> (Bs, 1)
            y_pred = self.pred(tokens)
        elif self.head_type == 'mlp':
            # Flatten -> (Bs, T*D) -> MLP -> (Bs, 1)
            y_pred = self.pred(tokens)
        elif self.head_type == 'noparam_mean':
            # Mean over both (1,2) at once
            y_pred = tokens.mean(dim=(1, 2), keepdim=True) 
            # Output is now (Bs, 1, 1), squeeze the last dim
            y_pred = y_pred.view(tokens.shape[0], 1)

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict