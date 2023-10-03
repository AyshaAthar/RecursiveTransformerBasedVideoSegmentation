import torch
import torch.nn as nn

import torch.nn.functional as F
import math

import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F
from segm_video.model.vivit.attention import Block
from segm_video.model.vivit.utils import init_weights


class Corrector(nn.Module):
    """
    Corrector module that fuses the information between the predicted embeddings and
    the embeddings encoded from the current input

    Args:
    -----
    n_layers: int
        Number of transformer layers in the transformer-based encoder
    d_model: int
        Dimensionality of the patch tokens used throughout the model
    d_ff: int
        Hidden dimension of the MLP in the transformer blocks
    n_heads: int
        Number of heads in the self-attention operations
    dropout: float
        Percentage of dropout used in the predictor transformer blocks.
    drop_path_rate: float
        Percentage of stochastic depth. 
    """

    def __init__(self, n_layers, d_model, d_ff, n_heads, dropout=0.0, drop_path_rate=0.1):
        """ Module initializer """        
        self.n_layers =  n_layers
        self.dropout = dropout
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        super().__init__()
                
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(
                dim=d_model,
                heads=n_heads,
                mlp_dim=d_ff,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(n_layers)])
        
        self.temporal_token = nn.Parameter(torch.randn(1, 2, 1 , d_model)*(d_model**(-0.5)))    
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.LayerNorm(d_model)
        self.apply(init_weights)
        return

        
    def forward(self, vit_embs, predictor_embds):
        """ 
        Fusing information from the predicted patch embeddings from the predictor module
        with those actual encoded embeddings coming from the transformer encoder.

        Args:
        -----
        vit_embs: torch Tensor
            Embeddings encoded from the current time step. Shape is (B, num_tokens, token_dim)
        predictor_embds: torch Tensor
            Predicted embeddings for the current time step. Shape is (B, num_tokens, token_dim)

        Returns:
        --------
        corrected_embds: torch tensor
            Result of fusing the information between the predicted and encoded embeddings.
            Shape is (B, num_tokens, token_dim)
        """
    
        B, num_tokens, _ = predictor_embds.shape
        device = predictor_embds.device
        
        num_tokens = vit_embs.shape[1]
        
        predictor_embds = predictor_embds.reshape(B,1,num_tokens,192)
        vit_embs = vit_embs.reshape(B,1,num_tokens,192)
    
        # Concatenating with vit_encoder output of current time step 
        corrector_input = torch.cat([predictor_embds, vit_embs], dim=1)   #(B,2,2305,192)

        temporal_token = self.temporal_token.repeat(B, 1, num_tokens, 1).to(device)
        corrector_input = corrector_input + temporal_token

        corrector_input = rearrange(corrector_input, "b t n d -> b (t n) d")
        
        # Passing through corrector transformer
        corrector_input = self.dropout(corrector_input)
        
        for blk in self.blocks:
            corrector_input = blk(corrector_input)
            
        corrected_embds = corrector_input[:, num_tokens:]  # keeping only last 'num_tokens
        return corrected_embds

                
