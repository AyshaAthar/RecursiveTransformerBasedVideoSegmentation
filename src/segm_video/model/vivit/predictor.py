import torch
import torch.nn as nn

import torch.nn.functional as F
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from pathlib import Path

import torch.nn.functional as F
from segm_video.model.vivit.attention import Block

from timm.models.layers import trunc_normal_
from segm_video.model.vivit.utils import init_weights


class Predictor(nn.Module):
    """
    Predictor module used to forecast future embedings conditioned
    on current and previous tokens

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
    
    MAX_TIME_STEPS = 100

    def __init__(self, n_layers, d_model, d_ff, n_heads, dropout=0.0, drop_path_rate=0.1):
        """ Module initializer """
        self.n_layers =  n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, Predictor.MAX_TIME_STEPS, 1, d_model)*(d_model**(-0.5)))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(
                dim=d_model,
                heads=n_heads,
                mlp_dim=d_ff,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(n_layers)])
        
        #self.norm = nn.LayerNorm(d_model)
        trunc_normal_(self.temporal_token, std=0.02)
        self.apply(init_weights)
        return
        
    def forward(self, x):
        """
        Predicting the patch tokens at the subsequent time step using
        a transformer predictor module.
        
        Args:
        -----
        x: torch Tensor
            Encoded patch embeddings from all images up to the current point.
            Shape is (B, num_time_steps_so_far, num_tokens_per_img, token_dim)
            
        Returns:
        --------
        predictor_output: torch Tensor
            Predicted tokens at the subsequent time step.
            Shape is (B, num_tokens_per_img, token_dim)
        """
        B, seq_len, num_tokens, _ = x.shape
        device = x.device     
        
        # processing and adding temporal positional encoding
        temporal_token = self.temporal_token[:, :seq_len]  # removing extra time steps
        temporal_token = temporal_token.repeat(B, 1, num_tokens, 1).to(device)

        pred_inputs = temporal_token + x

        # applying predictor transformer  TODO: add support for spatio-temporal predictor
        pred_inputs = rearrange(pred_inputs, "b t n d -> b (t n) d")
        
        pred_inputs = self.dropout(pred_inputs)
        
        for blk in self.blocks:
            pred_inputs = blk(pred_inputs)
        predictor_output = rearrange(pred_inputs , "b (t n) d  -> b t n d", t=seq_len)

        # keeping only final predicted embeddings and the forecast for the subsequent time step
        predictor_output = predictor_output[:, -1]

        return predictor_output