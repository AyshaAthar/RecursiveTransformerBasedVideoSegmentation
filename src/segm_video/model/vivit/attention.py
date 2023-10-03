import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from einops import rearrange

# from segm_model.model.vivit.utils import init_weights

class FeedForward(nn.Module):
    """
    MLP module used in the transformer blocks
    """

    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        return

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ 
    Self-attention operation
    """
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        return

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        """ 
        Forward pass through self-attention module
        
        Args:
        -----
        x: torch tensor
            Input tokens to process. Shape is (B, num_tokens, token_dim)
        mask: torch Tensor
            Never gets used
            
        Returns:
        --------
        x: torch tensor
            Output processed tokens. Shape is (B, num_tokens, token_dim)
        attn: torch tensor
            Attention maps resulting from key-query product + softmax
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    """
    Standard transformer encoder block with self-attention.
    """

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        """ """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        return

    def forward(self, x, mask=None, return_attention=False):
        """ """
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x








