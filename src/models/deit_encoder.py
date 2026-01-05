"""
DeiT Feature Extractor for semantic segmentation.

Wraps timm DeiT to extract multi-scale features compatible with UperNet.
Uses MultiLevelNeck (bilinear interpolation) to create pseudo-hierarchical 
features from the single-scale ViT/DeiT output.

This follows the approach described in the Swin Transformer paper (Table 3),
where they reference SETR [81] for constructing hierarchical features from DeiT.
SETR uses bilinear interpolation + convolutions (not ConvTranspose2d), which
is exactly what MultiLevelNeck implements.

Reference:
    - Swin Transformer paper: https://arxiv.org/abs/2103.14030 (Table 3)
    - SETR paper [81]: https://arxiv.org/abs/2012.15840
    - mmsegmentation: configs/vit/vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class MultiLevelNeck(nn.Module):
    """
    Multi-level neck that resizes features to different scales.
    
    This matches mmsegmentation's MultiLevelNeck:
    - Takes features all at the same resolution (1/16 for patch_size=16)
    - Resizes them to scales [4, 2, 1, 0.5] relative to input
    - Results in [1/4, 1/8, 1/16, 1/32] of original image
    
    Args:
        in_channels: Input channel dimension (same for all levels)
        out_channels: Output channel dimension (same for all levels)
        scales: Scale factors for each level [4, 2, 1, 0.5]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: Tuple[float, ...] = (4, 2, 1, 0.5),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        
        # Optional: 1x1 conv to change channel dimension if needed
        self.lateral_convs = nn.ModuleList()
        for _ in range(len(scales)):
            if in_channels != out_channels:
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
            else:
                self.lateral_convs.append(nn.Identity())
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Resize features to different scales.
        
        Args:
            features: List of [B, C, H, W] tensors, all at same resolution
        
        Returns:
            List of [B, out_channels, H*scale, W*scale] tensors
        """
        outputs = []
        for i, (feat, scale) in enumerate(zip(features, self.scales)):
            feat = self.lateral_convs[i](feat)
            
            if scale != 1:
                feat = F.interpolate(
                    feat,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False,
                )
            outputs.append(feat)
        
        return outputs


class DeiTFeatureExtractor(nn.Module):
    """
    DeiT backbone with MultiLevelNeck for hierarchical feature extraction.
    
    DeiT outputs single-scale features (all layers at 1/16 resolution).
    Following mmsegmentation's approach, we:
    1. Extract features from intermediate transformer layers (3, 6, 9, 12)
    2. Use bilinear interpolation (MultiLevelNeck) to create hierarchy
    3. Keep channel dimension the SAME across all levels (384 for DeiT-S)
    
    This matches the paper's 52M parameter count for DeiT-S + UperNet.
    
    Output feature scales (for 512x512 input with patch_size=16):
        - C1: [B, 384, 128, 128]   (1/4 scale via 4× bilinear upsample)
        - C2: [B, 384, 64, 64]     (1/8 scale via 2× bilinear upsample)
        - C3: [B, 384, 32, 32]     (1/16 scale, native resolution)
        - C4: [B, 384, 16, 16]     (1/32 scale via 0.5× bilinear downsample)
    
    Args:
        variant: DeiT model variant from timm
        pretrained: If True, load ImageNet pretrained weights
        img_size: Input image size (used for position embedding interpolation)
        use_gradient_checkpointing: If True, use gradient checkpointing
        extract_layers: Which transformer layers to extract features from
    """
    
    def __init__(
        self,
        variant: str = 'deit_small_patch16_224',
        pretrained: bool = True,
        img_size: int = 512,
        use_gradient_checkpointing: bool = False,
        extract_layers: Tuple[int, ...] = (2, 5, 8, 11),  # Layers 3, 6, 9, 12 (0-indexed)
    ):
        super().__init__()
        self.variant = variant
        self.img_size = img_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.extract_layers = extract_layers
        
        # Load DeiT from timm
        # Use .fb_in1k suffix for Facebook's ImageNet-1K pretrained weights
        model_name = variant
        if pretrained and not variant.endswith('.fb_in1k'):
            model_name = f"{variant}.fb_in1k"
        
        try:
            self.deit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                img_size=img_size,  # Handle position embedding interpolation
            )
        except Exception:
            # Fallback to base variant name
            self.deit = timm.create_model(
                variant,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
            )
        
        # Get DeiT's embedding dimension
        self.embed_dim = self.deit.embed_dim  # 384 for DeiT-S, 768 for DeiT-B
        self.patch_size = self.deit.patch_embed.patch_size[0]  # 16
        self.num_patches_side = img_size // self.patch_size  # 32 for 512/16
        
        # Output channels: ALL SAME (matching mmsegmentation approach)
        # This is key to matching the paper's 52M parameter count
        self.out_channels = [self.embed_dim] * 4  # [384, 384, 384, 384] for DeiT-S
        
        # MultiLevelNeck: bilinear interpolation to create hierarchy
        # scales=[4, 2, 1, 0.5] means: 4× up, 2× up, identity, 0.5× down
        self.neck = MultiLevelNeck(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,  # Keep same channels
            scales=(4, 2, 1, 0.5),
        )
        
        # For compatibility
        self._img_size = img_size
    
    @property
    def patches_resolution(self):
        """Return patch grid resolution for compatibility."""
        return [self.num_patches_side, self.num_patches_side]
    
    def set_img_size(self, img_size: int):
        """Set input image size for resolution calculations."""
        self._img_size = img_size
        self.num_patches_side = img_size // self.patch_size
    
    def _extract_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from intermediate transformer layers.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            List of features from specified layers, each [B, N, C]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.deit.patch_embed(x)  # [B, N, embed_dim]
        
        # Add cls token
        cls_token = self.deit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, embed_dim]
        
        # Add position embedding
        x = x + self.deit.pos_embed
        x = self.deit.pos_drop(x)
        
        # Extract features from specified layers
        features = []
        for i, block in enumerate(self.deit.blocks):
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
            if i in self.extract_layers:
                features.append(x)
        
        return features
    
    def _reshape_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape transformer features to spatial format.
        
        Args:
            x: Features [B, N+1, C] (including cls token)
        
        Returns:
            Spatial features [B, C, H, W]
        """
        B, N_plus_1, C = x.shape
        # Remove cls token
        x = x[:, 1:, :]  # [B, N, C]
        
        # Reshape to spatial
        H = W = self.num_patches_side
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        return_multi_scale: bool = False
    ) -> List[torch.Tensor]:
        """
        Extract multi-scale features from DeiT using MultiLevelNeck.
        
        Args:
            x: Input images [B, 3, H, W]
            return_multi_scale: If True, return features from all scales.
                               If False, return only final features.
        
        Returns:
            If return_multi_scale=True:
                List of features [c1, c2, c3, c4] at scales [1/4, 1/8, 1/16, 1/32]
            If return_multi_scale=False:
                Final features [B, embed_dim, H/32, W/32]
        """
        # Extract features from intermediate layers
        layer_features = self._extract_intermediate_features(x)
        
        # Reshape all to spatial format [B, C, H, W]
        # All at 1/16 resolution (patch_size=16)
        spatial_features = [self._reshape_to_spatial(f) for f in layer_features]
        
        # Apply MultiLevelNeck (bilinear interpolation) to create hierarchy
        # This matches mmsegmentation's approach exactly
        multi_scale_features = self.neck(spatial_features)
        
        if return_multi_scale:
            return multi_scale_features
        else:
            return multi_scale_features[-1]  # 1/32 scale
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class DeiTSegmentationWrapper(nn.Module):
    """
    Wrapper for DeiT encoder + segmentation head.
    
    Similar to ResNetSegmentationWrapper but for DeiT.
    
    Args:
        encoder: DeiTFeatureExtractor
        seg_head: Segmentation head (e.g., UperNetHead)
        freeze_encoder: If True, freeze encoder weights during training
    """
    
    def __init__(
        self,
        encoder: DeiTFeatureExtractor,
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
        # Extract multi-scale features from DeiT
        features_list = self.encoder(x, return_multi_scale=True)
        
        # Pass through segmentation head
        output = self.seg_head(features_list)
        
        return output
    
    def train(self, mode: bool = True):
        """Override train mode to keep encoder frozen if requested."""
        super().train(mode)
        
        if self.freeze_encoder:
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
