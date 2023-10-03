
from segm_video.model.vivit.utils import init_weights
import torch
import torch.nn as nn

import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from segm_video.model.vivit.attention import Block
from segm_video.model.vivit.predictor import Predictor
from segm_video.model.vivit.corrector import Corrector
from segm_video.model.vivit.decoder import Decoder
from segm_video.model.vivit.vit import Vit
from segm_video.model.vivit.utils import init_weights
from segm_video.model.utils import padding, unpadding


class ViViT(nn.Module):
    """
    Video Extension of the segmenter model for video semantic segmentation

    Args:
    -----
    image_size: tuple/iterable
        Sizes of the images input to the model
    patch_size: int
        Size of the image patches to extract from the images.
    n_layers: int
        Number of transformer layers in the transformer-based encoder
    n_layers_pred: int
        Number of transformer layers in the transformer-based predictor
    n_layers_corr: int
        Number of transformer layers in the transformer-based corrector        
    d_model: int
        Dimensionality of the patch tokens used throughout the model
    d_ff: int
        Hidden dimension of the MLP in the transformer blocks
    n_heads: int
        Number of heads in the self-attention operations
    n_cls: int
        Number of classes in the semantic segmentation task
    dropout: float
        Percentage of dropout. 
    drop_path_rate: float
        Percentage of stochastic depth
    distilled: bool
        Selects the backbone ViT vs DeiT
    channels: int
        Number of input channels, which defaults to 3
    """

    def __init__(self, image_size, patch_size, n_layers,n_layers_pred,n_layers_corr, d_model, d_ff, n_heads, n_cls,decoder_type,
                 dropout=0.1, drop_path_rate=0.0, distilled=False, channels=3):
        """ Video Segmenter initializer """
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers =  n_layers
        self.n_layers_pred = n_layers_pred 
        self.n_layers_corr = n_layers_corr
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_cls = n_cls
        self.d_ff = d_ff
        super().__init__()
 
        self.vit_embs = Vit(
            image_size=image_size,
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            n_cls=n_cls,
            n_heads=n_heads,
            d_ff=d_ff
        )        
        self.predictor_embs = Predictor(
            n_layers=n_layers_pred,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads
        )
        self.corrector_embs = Corrector(
            n_layers=n_layers_corr,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads
        )
        self.decoder = Decoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            n_cls=n_cls,
            decoder_type=decoder_type
        )
        self.apply(init_weights)
        return
    
    def forward(self, x):
        """
        Forward pass through the video segmenter model
        
        Args:
        -----
        x: torch Tensor
            Sequence of images to segment. Shape is (B, T, C, H, W)
        
        Returns:
        --------
        segmentations: torch Tensor
            Semantic segmentation of each of the input frames.
            Shape is (B, T, num_classes, H, W)
        vit_embds: torch Tensor
            Encoded patch embeddings for each of the input frames.
            Shape is (B, T, num_patches, token_dim)
        predictor_embds: torch Tensor
            Predicted next-step patch embeddings
            Shape is (B, (T-1), num_patches, token_dim)
        corrector_embds: torch Tensor
            Fusion between predicted patch embeddings and encoded embeddings.
            Shape is (B, (T-1), num_patches, token_dim)
        """
        seq_len = x.shape[1]
        
        b, t, c, h, w = x.shape
        x = padding(x, self.patch_size)

        # encoding all images into patch embeddings
        vit_embds = self.vit_embs(x)
        segmentations, predictor_embds, corrector_embds = [], [], []
        for i in range(0, seq_len):
            if i > 0:
                # predicting future patch embeddings and performing correction step
                if i>4:
                    cur_predictor_embds = self.predictor_embs(vit_embds[:, i-4:i])
                else:  
                    cur_predictor_embds = self.predictor_embs(vit_embds[:, :i])
                cur_corrector_embds = self.corrector_embs(
                    vit_embs=vit_embds[:, i],
                    predictor_embds=cur_predictor_embds
                )
                predictor_embds.append(cur_predictor_embds)
                corrector_embds.append(cur_corrector_embds)
            # decoding fused patch embeddings or encoded embeddings
            embs_to_decode = cur_corrector_embds if i > 0 else vit_embds[:, 0]

            embs_to_decode = embs_to_decode[:, 1:]

            segmentation = self.decoder(patch_embs=embs_to_decode)

            segmentation = unpadding(segmentation, (h,w))
            segmentations.append(segmentation)
     
        predictor_embds = torch.stack(predictor_embds, dim=1)
        corrector_embds = torch.stack(corrector_embds, dim=1)
        segmentations = torch.stack(segmentations, dim=1)
        return segmentations, vit_embds, predictor_embds,corrector_embds
        