import torch
import torch.nn as nn

import torch.nn.functional as F
import math
from collections import defaultdict

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights

from einops import rearrange

from segm_video.model.vivit.attention import Block
from segm_video.model.vivit.utils import init_weights


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    """
    Rescaling the grid of position embeddings when loading from state_dict.

    Adapted from:
    https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


class PatchEmbedding(nn.Module):
    """
    Module used to embed image patches into patch embeddings using a convolutional layer
    """

    def __init__(self, image_size, patch_size, embed_dim, channels):
        """ Module initializer """
        super().__init__()
        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError(f"{image_size =} must be divisible by the {patch_size =}")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, im):
        """
        Embedding image patches.
        The process is as follows:
          - (B, 3, H, W)  --> (B, patch_dim, n_patches_H, n_patches_W)
          - (B, patch_dim, n_patches_H, n_patches_W) --> (B, patch_dim, n_patches) 
          - (B, patch_dim, n_patches) --> (B, n_patches, patch_dim)
        
        Args:
        -----
        im: torch Tensor
            Image to split into patches and embedd. Shape is (B, 3, H, W)
        
        Returns:
        --------
        patch_embs: torch Tensor
            Patch embeddings. Shape is (B, num_patches, patch_dim)
        """
        patch_embs = self.proj(im).flatten(2).transpose(1, 2)
        return patch_embs


class VisionTransformer(nn.Module):
    """
    Vision transformer (ViT or DeiT) encoder module
    """

    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        n_cls,
        n_heads,
        d_ff,
        dropout=0.0,
        drop_path_rate=0.1,
        distilled=False,
        channels=3
    ):
  
        """ Encoder initializer """
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=d_model,
            channels=channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.distilled = distilled

        # cls and pos tokens
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 2, d_model))
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(
                dim=d_model,
                heads=n_heads,
                mlp_dim=d_ff,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        #self.head = nn.Linear(d_model, n_cls)
        
        # initialization
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)
        return

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False):
        """ 
        Encoding an image by projecting it into patch embeddings, and processing
        these with a transformer encoder
        
        Args:
        -----
        im: torch tensor
            Input image. Shape is (B, C, H, W)
        return_features: bool
            Never really used
        """
        B, _, H, W = im.shape
        PS = self.patch_size

        # projecting into patch embedding
        x = self.patch_embed(im)  # (B, num_patch_embs, patch_emb_dim)

        # concatenating the extra tokens (class and distilled) with the patch embeddings 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # adapting the positional embeddings in case shapes do not match pretraining or if params change.
        pos_embed = self.pos_embed
        self.num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                posemb=pos_embed,
                grid_old_shape=self.patch_embed.grid_size,
                grid_new_shape=(H // PS, W // PS),
                num_extra_tokens=self.num_extra_tokens,
            )
            
        # adding positional embedding and encoding embeddings via transformer
        x = x + pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  

        return x

    def get_attention_map(self, im, layer_id):
        """
        Fetching attention maps from a particular transformer encoder layer.
        
        Args:
        -----
        im: torch tensor
            Input image. Shape is (B, C, H, W)
        layer_id: int
            Index of the transformer encoder layer to extract the attention maps from
        """
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(f"Provided {layer_id = } is not valid. 0 <= {layer_id} < {self.n_layers}.")
        B, _, H, W = im.shape
        PS = self.patch_size

        # patch embedding and adding extra tokens
        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                posemb=pos_embed,
                grid_old_shape=self.patch_embed.grid_size,
                grid_new_shape=(H // PS, W // PS),
                num_extra_tokens=num_extra_tokens,
            )
        x = x + pos_embed

        # encoding and returning attention maps from desired layer
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
        return


class Vit(nn.Module):
    """ 
    Transformer encoder module used to map a sequence of images into their corresponding
    patch embeddings.
    """
    
    def __init__(self, image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls,
                 dropout=0.0, drop_path_rate=0.1, distilled=False, channels=3):
        """ Module initializer """    
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers =  n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_cls = n_cls
        self.d_ff = d_ff
        super().__init__()
        
        self.encoder = VisionTransformer(
            image_size=image_size, 
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            n_cls=n_cls,
            n_heads=n_heads,
            d_ff=d_ff
        )
        return                
        
    def forward(self, images):
        """
        Mapping a sequence of images into their corresponding patch embeddings

        Args:
        -----
        images: torch tensor
            Images to embed into patch embeddings. Shape is (B, T, C, H, W)

        Returns:
        --------
        patch_embeddings: torch Tensor
            Patch embeddings from every image in the input sequence.
            Shape is (B, T, num_patches, patch_dim)
        """
        num_seqs = images.shape[1]
        vit_transformer_out = []
        for i in range(num_seqs):
            x = self.encoder(images[:, i])
            vit_transformer_out.append(x)
        patch_embeddings = torch.stack(vit_transformer_out, dim=1)
        return patch_embeddings