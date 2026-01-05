"""
DeiT Feature Extractor for semantic segmentation.

Wraps timm DeiT to extract multi-scale features compatible with UperNet.
Uses deconvolution layers to create pseudo-hierarchical features from
the single-scale ViT/DeiT output, following the approach described in
the Swin Transformer paper (Table 3, footnote †).

Reference:
    "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/abs/2103.14030
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


class DeiTFeatureExtractor(nn.Module):
    """
    DeiT backbone with deconvolution layers for hierarchical feature extraction.
    
    DeiT naturally outputs single-scale features (all layers at 1/16 resolution).
    To make it compatible with UperNet (which expects multi-scale features),
    we extract features from intermediate transformer layers and use 
    deconvolution to create a pseudo-hierarchy.
    
    Following the Swin paper's approach:
    - Extract features from layers 3, 6, 9, 12 (for 12-layer DeiT)
    - Use deconvolution to upsample to different scales
    - Match the scale/channel pattern expected by UperNet
    
    Output feature scales (for 512x512 input):
        - C1: [B, 96, 128, 128]   (1/4 scale via 4× deconv)
        - C2: [B, 192, 64, 64]    (1/8 scale via 2× deconv)
        - C3: [B, 384, 32, 32]    (1/16 scale, native resolution)
        - C4: [B, 768, 16, 16]    (1/32 scale via 2× downsample)
    
    Args:
        variant: DeiT model variant from timm
        pretrained: If True, load ImageNet pretrained weights
        img_size: Input image size (used for position embedding interpolation)
        use_gradient_checkpointing: If True, use gradient checkpointing
        extract_layers: Which transformer layers to extract features from
    """
    
    # Channel configuration to match Swin-T style hierarchy
    # Input: DeiT-S embed_dim=384
    # Output: [96, 192, 384, 768] to match Swin-T pattern
    OUT_CHANNELS = [96, 192, 384, 768]
    
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
            # Try the Facebook pretrained version first
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
        
        # Store output channels for UperNet
        self.out_channels = self.OUT_CHANNELS
        
        # Build deconvolution layers to create hierarchical features
        # DeiT-S has embed_dim=384, we create hierarchy [96, 192, 384, 768]
        self._build_deconv_layers()
        
        # For compatibility
        self._img_size = img_size
    
    def _build_deconv_layers(self):
        """
        Build deconvolution and projection layers to create hierarchical features.
        
        From DeiT features at 1/16 scale, we create:
        - C1: 1/4 scale (4× upsample)
        - C2: 1/8 scale (2× upsample)
        - C3: 1/16 scale (identity)
        - C4: 1/32 scale (2× downsample)
        """
        embed_dim = self.embed_dim  # 384 for DeiT-S
        
        # C1: 4× upsample (1/16 → 1/4)
        # Use transposed convolution for learnable upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.ConvTranspose2d(192, self.OUT_CHANNELS[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(self.OUT_CHANNELS[0]),
            nn.GELU(),
        )
        
        # C2: 2× upsample (1/16 → 1/8)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, self.OUT_CHANNELS[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(self.OUT_CHANNELS[1]),
            nn.GELU(),
        )
        
        # C3: Identity projection (1/16 → 1/16)
        self.lateral3 = nn.Sequential(
            nn.Conv2d(embed_dim, self.OUT_CHANNELS[2], kernel_size=1),
            nn.BatchNorm2d(self.OUT_CHANNELS[2]),
            nn.GELU(),
        )
        
        # C4: 2× downsample (1/16 → 1/32)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(embed_dim, self.OUT_CHANNELS[3], kernel_size=2, stride=2),
            nn.BatchNorm2d(self.OUT_CHANNELS[3]),
            nn.GELU(),
        )
    
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
        Extract multi-scale features from DeiT using deconvolution.
        
        Args:
            x: Input images [B, 3, H, W]
            return_multi_scale: If True, return features from all scales.
                               If False, return only final features.
        
        Returns:
            If return_multi_scale=True:
                List of features [c1, c2, c3, c4] with shapes matching UperNet expectations
            If return_multi_scale=False:
                Final features [B, 768, H/32, W/32]
        """
        # Extract features from intermediate layers
        layer_features = self._extract_intermediate_features(x)
        
        # Reshape all to spatial format [B, C, H, W]
        # All at 1/16 resolution (patch_size=16)
        spatial_features = [self._reshape_to_spatial(f) for f in layer_features]
        
        # Apply deconvolution to create hierarchy
        # Use features from different layers for diversity
        c1 = self.deconv1(spatial_features[0])      # Layer 3 → 4× upsample → 1/4
        c2 = self.deconv2(spatial_features[1])      # Layer 6 → 2× upsample → 1/8
        c3 = self.lateral3(spatial_features[2])     # Layer 9 → identity → 1/16
        c4 = self.downsample4(spatial_features[3])  # Layer 12 → 2× downsample → 1/32
        
        if return_multi_scale:
            return [c1, c2, c3, c4]
        else:
            return c4
    
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
