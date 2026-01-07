"""
Segmentation model wrapper for combining encoder + segmentation head.

Similar to ModelWrapper for classification, but designed for segmentation tasks.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SegmentationModelWrapper(nn.Module):
    """
    Wrapper for segmentation models combining an encoder and segmentation head.
    
    Architecture:
        Input Image → Encoder (multi-scale features) → Segmentation Head → Output
    
    Args:
        encoder: Feature extractor (e.g., SwinTransformerModel)
        seg_head: Segmentation head (e.g., UperNetHead)
        freeze_encoder: If True, freeze encoder weights during training
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        seg_head: nn.Module,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.seg_head = seg_head
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and segmentation head.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Extract multi-scale features from encoder
        # For Swin-T: returns tuples of [B, C, H, W] tensors
        features = self.encoder(x)
        
        # Pass through segmentation head
        output = self.seg_head(features)
        
        return output
    
    def train(self, mode: bool = True):
        """Override train mode to keep encoder frozen if requested."""
        super().train(mode)
        
        if self.freeze_encoder:
            # Keep encoder in eval mode even when wrapper is in train mode
            self.encoder.eval()
        
        return self
    
    def get_num_params(self) -> dict:
        """Get parameter counts for encoder and head separately."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.seg_head.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "encoder": encoder_params,
            "head": head_params,
            "total": encoder_params + head_params,
            "trainable": trainable_params,
        }
