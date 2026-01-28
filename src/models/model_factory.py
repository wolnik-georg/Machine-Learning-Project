"""
Model factory for creating different architectures for comparison experiments.
"""

import torch
from torch.utils.checkpoint import checkpoint
import timm
import torchvision.models as models
from .swin.swin_transformer_model import SwinTransformerModel
from .model_wrapper import ModelWrapper
from .segmentation_wrapper import SegmentationModelWrapper
from .resnet_encoder import ResNetFeatureExtractor, ResNetSegmentationWrapper
from .deit_encoder import DeiTFeatureExtractor, DeiTSegmentationWrapper
from .heads import LinearClassificationHead, UperNetHead


def create_model(config):
    """
    Create model based on configuration type.

    Args:
        config: MODEL_CONFIG dictionary

    Returns:
        ModelWrapper containing the model
    """
    model_type = config["type"]

    if model_type == "swin":
        return create_swin_model(config)
    elif model_type == "swin_hybrid":
        return create_swin_hybrid_model(config)
    elif model_type == "swin_improved":
        return create_swin_improved_model(config)
    elif model_type == "vit":
        return create_vit_model(config)
    elif model_type == "resnet":
        return create_resnet_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_swin_model(config):
    """Create Swin Transformer model (existing implementation)."""
    encoder = SwinTransformerModel(
        img_size=224,  # Fixed for ImageNet
        patch_size=config["patch_size"],
        embedding_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        dropout_rate=config["dropout"],
        attention_dropout_rate=config["attention_dropout"],
        projection_dropout_rate=config["projection_dropout"],
        drop_path_rate=config["drop_path_rate"],
        use_shifted_window=config["use_shifted_window"],
        use_relative_bias=config["use_relative_bias"],
        use_absolute_pos_embed=config["use_absolute_pos_embed"],
        use_hierarchical_merge=config["use_hierarchical_merge"],
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
    )

    pred_head = LinearClassificationHead(
        num_features=encoder.num_features,
        num_classes=1000,  # ImageNet
    )

    return ModelWrapper(
        encoder=encoder,
        pred_head=pred_head,
        freeze=False,  # From scratch training
    )


def create_swin_hybrid_model(config):
    """Create Hybrid CNN-Swin model with early fusion."""
    # Create the base Swin model
    swin_encoder = SwinTransformerModel(
        img_size=224,  # Fixed for ImageNet
        patch_size=config["patch_size"],
        embedding_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        dropout_rate=config["dropout"],
        attention_dropout_rate=config["attention_dropout"],
        projection_dropout_rate=config["projection_dropout"],
        drop_path_rate=config["drop_path_rate"],
        use_shifted_window=config["use_shifted_window"],
        use_relative_bias=config["use_relative_bias"],
        use_absolute_pos_embed=config["use_absolute_pos_embed"],
        use_hierarchical_merge=config["use_hierarchical_merge"],
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
    )

    # Create CNN stem for early fusion
    cnn_stem_config = config.get("cnn_stem_config", {})
    cnn_stem = create_cnn_stem(
        in_channels=3,
        embed_dim=config["embed_dim"],
        channels=cnn_stem_config.get("channels", [32, 64]),
        kernel_size=cnn_stem_config.get("kernel_size", 3),
        stride=cnn_stem_config.get("stride", 2),
        padding=cnn_stem_config.get("padding", 1),
        activation=cnn_stem_config.get("activation", "gelu"),
        use_batch_norm=cnn_stem_config.get("use_batch_norm", True),
    )

    # Create hybrid encoder
    hybrid_encoder = HybridSwinEncoder(
        swin_model=swin_encoder,
        cnn_stem=cnn_stem,
        use_cnn_stem=config.get("use_cnn_stem", True),
        embed_dim=config["embed_dim"],
    )

    pred_head = LinearClassificationHead(
        num_features=hybrid_encoder.num_features,
        num_classes=1000,  # ImageNet
    )

    return ModelWrapper(
        encoder=hybrid_encoder,
        pred_head=pred_head,
        freeze=False,  # From scratch training
    )


def create_cnn_stem(
    in_channels,
    embed_dim,
    channels,
    kernel_size,
    stride,
    padding,
    activation,
    use_batch_norm,
):
    """Create lightweight CNN stem for early fusion."""
    layers = []
    current_channels = in_channels

    # Intermediate convolutional layers
    for out_channels in channels:
        layers.extend(
            [
                torch.nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            ]
        )
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm2d(out_channels))

        if activation == "gelu":
            layers.append(torch.nn.GELU())
        elif activation == "silu":
            layers.append(torch.nn.SiLU())
        else:
            layers.append(torch.nn.ReLU())

        current_channels = out_channels

    # Final projection to match Swin embed_dim
    layers.extend(
        [
            torch.nn.Conv2d(
                current_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        ]
    )
    if use_batch_norm:
        layers.append(torch.nn.BatchNorm2d(embed_dim))

    if activation == "gelu":
        layers.append(torch.nn.GELU())
    elif activation == "silu":
        layers.append(torch.nn.SiLU())
    else:
        layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


class HybridSwinEncoder(torch.nn.Module):
    """Hybrid CNN-Swin encoder with early fusion."""

    def __init__(self, swin_model, cnn_stem, use_cnn_stem, embed_dim):
        super().__init__()
        self.swin_model = swin_model
        self.cnn_stem = cnn_stem
        self.use_cnn_stem = use_cnn_stem
        self.embed_dim = embed_dim

        # Copy important attributes from swin model
        self.num_features = swin_model.num_features
        self.patch_embed = swin_model.patch_embed if not use_cnn_stem else None

    def forward(self, x):
        if self.use_cnn_stem:
            # Apply CNN stem
            x = self.cnn_stem(x)  # → B, C, H/8, W/8

            # Flatten spatial dimensions to create patch tokens
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # → B, HW/64, C

            # Add absolute position embedding if the original Swin uses it
            if (
                hasattr(self.swin_model, "absolute_pos_embed")
                and self.swin_model.absolute_pos_embed is not None
            ):
                x = x + self.swin_model.absolute_pos_embed
        else:
            # Fallback to vanilla patch embedding
            x, (H, W) = self.patch_embed(x)

        # Proceed with Swin transformer layers
        for layer in self.swin_model.layers:
            x, H, W = layer(x, H, W)

        # Return token sequence [B, L, C] (same format as SwinTransformerModel.forward_features)
        return x


def create_swin_improved_model(config):
    """Create Improved Swin model with conv stem and inverted residual FFN."""
    # Create the base Swin model
    swin_encoder = SwinTransformerModel(
        img_size=224,  # Fixed for ImageNet
        patch_size=config["patch_size"],
        embedding_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        dropout_rate=config["dropout"],
        attention_dropout_rate=config["attention_dropout"],
        projection_dropout_rate=config["projection_dropout"],
        drop_path_rate=config["drop_path_rate"],
        use_shifted_window=config["use_shifted_window"],
        use_relative_bias=config["use_relative_bias"],
        use_absolute_pos_embed=config["use_absolute_pos_embed"],
        use_hierarchical_merge=config["use_hierarchical_merge"],
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
    )

    # Create convolutional stem
    conv_stem_config = config.get("conv_stem_config", {})
    conv_stem = create_conv_stem(
        in_channels=3,
        embed_dim=config["embed_dim"],
        channels=conv_stem_config.get("channels", [48, 96]),
        kernel_sizes=conv_stem_config.get("kernel_sizes", [4, 3]),
        strides=conv_stem_config.get("strides", [4, 2]),
        paddings=conv_stem_config.get("paddings", [0, 1]),
        activation=conv_stem_config.get("activation", "gelu"),
        use_batch_norm=conv_stem_config.get("use_batch_norm", True),
    )

    # Create improved encoder with conv stem and inverted FFN
    improved_encoder = ImprovedSwinEncoder(
        swin_model=swin_encoder,
        conv_stem=conv_stem,
        use_conv_stem=config.get("use_conv_stem", True),
        embed_dim=config["embed_dim"],
        ffn_config=config.get("ffn_config", {}),
        use_inverted_ffn=config.get("use_inverted_ffn", True),
    )

    pred_head = LinearClassificationHead(
        num_features=improved_encoder.num_features,
        num_classes=1000,  # ImageNet
    )

    return ModelWrapper(
        encoder=improved_encoder,
        pred_head=pred_head,
        freeze=False,  # From scratch training
    )


def create_conv_stem(
    in_channels,
    embed_dim,
    channels,
    kernel_sizes,
    strides,
    paddings,
    activation,
    use_batch_norm,
):
    """Create overlapping convolutional stem for improved patch embedding."""
    layers = []
    current_channels = in_channels

    # Intermediate convolutional layers
    for i, (out_channels, kernel_size, stride, padding) in enumerate(
        zip(channels, kernel_sizes, strides, paddings)
    ):
        layers.extend(
            [
                torch.nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            ]
        )
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm2d(out_channels))

        if activation == "gelu":
            layers.append(torch.nn.GELU())
        elif activation == "silu":
            layers.append(torch.nn.SiLU())
        else:
            layers.append(torch.nn.ReLU())

        current_channels = out_channels

    # Final projection to match Swin embed_dim
    layers.extend(
        [
            torch.nn.Conv2d(
                current_channels, embed_dim, kernel_size=1, stride=1, padding=0
            ),
        ]
    )
    if use_batch_norm:
        layers.append(torch.nn.BatchNorm2d(embed_dim))

    if activation == "gelu":
        layers.append(torch.nn.GELU())
    elif activation == "silu":
        layers.append(torch.nn.SiLU())
    else:
        layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


class InvertedResidualFFN(torch.nn.Module):
    """Inverted Residual FFN with depthwise convolution (MobileNetV2 style)."""

    def __init__(self, dim, expand_ratio=4, drop=0.0, activation="gelu"):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)

        # Expansion
        self.fc1 = torch.nn.Linear(dim, hidden_dim)

        # Depthwise convolution for local mixing
        self.dwconv = torch.nn.Conv2d(
            hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim
        )

        # Activation
        if activation == "gelu":
            self.act = torch.nn.GELU()
        elif activation == "silu":
            self.act = torch.nn.SiLU()
        else:
            self.act = torch.nn.ReLU()

        # Projection back
        self.fc2 = torch.nn.Linear(hidden_dim, dim)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        """
        Forward accepts token sequence x with shape [B, L, C].
        We infer spatial H, W from L (sqrt(L)) so the depthwise
        convolution can be applied. This matches the call signature of
        the original MLP used in Swin blocks.
        """
        import math

        # Expansion
        x = self.fc1(x)
        x = self.act(x)  # Apply activation BEFORE depthwise conv

        # Infer spatial dims from sequence length
        B, L, C = x.shape
        s = int(math.isqrt(L))
        if s * s == L:
            H = W = s
        else:
            # fallback: find a factorization close to sqrt(L)
            H = None
            for candidate in range(s, 0, -1):
                if L % candidate == 0:
                    H = candidate
                    W = L // candidate
                    break
            if H is None:
                # As a last resort, treat sequence as H=1, W=L
                H, W = 1, L

        # Reshape for depthwise conv: B, L, C -> B, C, H, W
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        # Depthwise convolution
        # Force fp32 for depthwise conv to avoid cuDNN issues with bf16 grouped convolutions
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float() if x.dtype == torch.bfloat16 else x
            x = self.dwconv(x)
            x = x.to(torch.bfloat16) if x.dtype == torch.float32 else x

        # Reshape back: B, C, H, W -> B, L, C
        x = x.flatten(2).transpose(1, 2).contiguous()

        # Projection back to original dimension
        x = self.fc2(x)
        x = self.drop(x)

        # NO residual here - it's handled at the block level in SwinTransformerBlock
        return x


class ImprovedSwinEncoder(torch.nn.Module):
    """Improved Swin encoder with conv stem and inverted residual FFN."""

    def __init__(
        self,
        swin_model,
        conv_stem,
        use_conv_stem,
        embed_dim,
        ffn_config,
        use_inverted_ffn,
    ):
        super().__init__()
        self.swin_model = swin_model
        self.conv_stem = conv_stem
        self.use_conv_stem = use_conv_stem
        self.embed_dim = embed_dim
        self.use_inverted_ffn = use_inverted_ffn

        # Copy important attributes from swin model
        self.num_features = swin_model.num_features

        # Replace FFN blocks if using inverted residual
        if use_inverted_ffn:
            self._replace_ffn_blocks(ffn_config)

    def _replace_ffn_blocks(self, ffn_config):
        """Replace all FFN blocks in Swin layers with inverted residual FFN."""
        expand_ratio = ffn_config.get("expand_ratio", 4)
        activation = ffn_config.get("activation", "gelu")

        for layer in self.swin_model.layers:
            for block in layer.blocks:
                # Replace the MLP with inverted residual FFN
                original_mlp = block.mlp
                # Infer input dim from original linear layer
                try:
                    in_dim = original_mlp.fc1.in_features
                except Exception:
                    in_dim = self.embed_dim
                # Try to get dropout probability if available
                try:
                    drop_p = getattr(original_mlp.drop1, "p", 0.0)
                except Exception:
                    drop_p = 0.0

                inverted_ffn = InvertedResidualFFN(
                    dim=in_dim,
                    expand_ratio=expand_ratio,
                    drop=drop_p,
                    activation=activation,
                )
                block.mlp = inverted_ffn

    def forward(self, x):
        if self.use_conv_stem:
            # Apply convolutional stem
            x = self.conv_stem(x)  # → B, C, H/4, W/4

            # Flatten spatial dimensions to create patch tokens
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # → B, HW/16, C

            # Add absolute position embedding if the original Swin uses it
            if (
                hasattr(self.swin_model, "absolute_pos_embed")
                and self.swin_model.absolute_pos_embed is not None
            ):
                x = x + self.swin_model.absolute_pos_embed
        else:
            # Fallback to vanilla patch embedding
            x, (H, W) = self.swin_model.patch_embed(x)

        # Proceed with Swin transformer layers
        for layer in self.swin_model.layers:
            x, H, W = layer(x, H, W)

        # Return token sequence [B, L, C] (same format as SwinTransformerModel.forward_features)
        return x


def create_vit_model(config):
    """Create ViT model using timm library with gradient checkpointing."""
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=config["num_classes"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_path_rate=0.1,  # Add some dropout for regularization
    )

    # Enable gradient checkpointing for memory efficiency if requested
    if config.get("use_gradient_checkpointing", False):
        if hasattr(model, "grad_checkpointing"):
            model.grad_checkpointing = True
        elif hasattr(model, "blocks"):
            # Manually enable gradient checkpointing for blocks
            original_forward = model.blocks.forward

            def checkpointed_forward(x):
                if model.training:
                    for block in model.blocks:
                        x = checkpoint(block, x, use_reentrant=False)
                    return x
                else:
                    return original_forward(x)

            model.blocks.forward = checkpointed_forward

    return model


def create_resnet_model(config):
    """Create ResNet model using torchvision with gradient checkpointing."""
    # Use layers config to choose ResNet variant
    layers = config.get("layers", [3, 4, 6, 3])

    if layers == [3, 4, 6, 3]:
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])
    elif layers == [3, 4, 23, 3]:
        model = models.resnet101(pretrained=False, num_classes=config["num_classes"])
    else:
        # Default to ResNet-50
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])

    # Enable gradient checkpointing for memory efficiency if requested
    if config.get("use_gradient_checkpointing", False):
        if (
            hasattr(model, "layer1")
            and hasattr(model, "layer2")
            and hasattr(model, "layer3")
            and hasattr(model, "layer4")
        ):
            # Wrap each ResNet layer with gradient checkpointing
            original_layer1_forward = model.layer1.forward
            original_layer2_forward = model.layer2.forward
            original_layer3_forward = model.layer3.forward
            original_layer4_forward = model.layer4.forward

            def checkpointed_layer1(x):
                return (
                    checkpoint(original_layer1_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer1_forward(x)
                )

            def checkpointed_layer2(x):
                return (
                    checkpoint(original_layer2_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer2_forward(x)
                )

            def checkpointed_layer3(x):
                return (
                    checkpoint(original_layer3_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer3_forward(x)
                )

            def checkpointed_layer4(x):
                return (
                    checkpoint(original_layer4_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer4_forward(x)
                )

            model.layer1.forward = checkpointed_layer1
            model.layer2.forward = checkpointed_layer2
            model.layer3.forward = checkpointed_layer3
            model.layer4.forward = checkpointed_layer4

    return model


def create_segmentation_model(swin_config, downstream_config, load_pretrained=True):
    """
    Create segmentation model with Swin encoder + UperNet head.

    Args:
        swin_config: SWIN_CONFIG dictionary from config
        downstream_config: DOWNSTREAM_CONFIG dictionary from config
        load_pretrained: If True, load ImageNet pretrained weights for encoder

    Returns:
        SegmentationModelWrapper containing encoder + segmentation head
    """
    from src.utils.load_weights import load_pretrained_reference, transfer_weights

    # Create Swin Transformer encoder
    encoder = SwinTransformerModel(
        img_size=swin_config["img_size"],  # 512 for ADE20K
        patch_size=swin_config["patch_size"],
        embedding_dim=swin_config["embed_dim"],
        depths=swin_config["depths"],
        num_heads=swin_config["num_heads"],
        window_size=swin_config["window_size"],
        mlp_ratio=swin_config["mlp_ratio"],
        dropout_rate=swin_config["dropout"],
        out_indices=swin_config["out_indices"],
        attention_dropout_rate=swin_config["attention_dropout"],
        projection_dropout_rate=swin_config["projection_dropout"],
        drop_path_rate=swin_config["drop_path_rate"],
        use_shifted_window=swin_config.get("use_shifted_window", True),
        use_relative_bias=swin_config.get("use_relative_bias", True),
        use_absolute_pos_embed=swin_config.get("use_absolute_pos_embed", False),
        use_hierarchical_merge=swin_config.get("use_hierarchical_merge", False),
        use_gradient_checkpointing=swin_config.get("use_gradient_checkpointing", False),
    )

    # Calculate in_channels for each stage
    # Swin-T: [96, 192, 384, 768]
    embed_dim = swin_config["embed_dim"]
    in_channels = [int(embed_dim * (2**i)) for i in range(len(swin_config["depths"]))]

    # Create UperNet segmentation head
    seg_head = UperNetHead(
        in_channels=in_channels,
        num_classes=downstream_config["num_classes"],
        channels=512,  # FPN channels (standard for UperNet)
        pool_scales=(1, 2, 3, 6),  # PPM scales from paper
        dropout=0.1,  # Dropout before classifier
    )

    # Combine encoder + head
    model = SegmentationModelWrapper(
        encoder=encoder,
        seg_head=seg_head,
        freeze_encoder=downstream_config.get("freeze_encoder", False),
    )

    # Load pretrained ImageNet weights for encoder
    if load_pretrained and downstream_config.get("use_pretrained", True):
        pretrained_model_name = "swin_tiny_patch4_window7_224"  # TIMM model name
        print(f"Loading pretrained weights from TIMM: {pretrained_model_name}")

        pretrained_model = load_pretrained_reference(
            model_name=pretrained_model_name,
            device="cpu",  # Load to CPU first, then move to device
        )

        if pretrained_model is not None:
            stats = transfer_weights(
                custom_model=model,
                pretrained_model=pretrained_model,
                encoder_only=True,  # Only transfer encoder weights
            )
            print(
                f"Weight transfer complete: {stats['transferred']} layers transferred, "
                f"{stats['missing']} missing, {stats['size_mismatches']} size mismatches"
            )
        else:
            print("Warning: Could not load pretrained weights. Training from scratch.")

    return model


def create_resnet_segmentation_model(resnet_config, downstream_config):
    """
    Create segmentation model with ResNet encoder + UperNet head.

    This function loads a pretrained ResNet-101/50 from torchvision and combines
    it with UperNet for semantic segmentation on ADE20K or similar datasets.

    Args:
        resnet_config: ResNet configuration dictionary with keys:
            - variant: 'resnet50' or 'resnet101' (default: 'resnet101')
            - pretrained: Whether to load ImageNet pretrained weights (default: True)
            - img_size: Input image size (default: 512)
        downstream_config: DOWNSTREAM_CONFIG dictionary with keys:
            - num_classes: Number of segmentation classes (e.g., 150 for ADE20K)
            - freeze_encoder: If True, freeze ResNet encoder weights

    Returns:
        ResNetSegmentationWrapper containing encoder + segmentation head
    """
    # Get ResNet variant
    variant = resnet_config.get("variant", "resnet101")
    pretrained = resnet_config.get("pretrained", True)
    img_size = resnet_config.get("img_size", 512)
    use_gradient_checkpointing = resnet_config.get("use_gradient_checkpointing", False)

    print(
        f"Creating ResNet segmentation model: {variant}, pretrained={pretrained}, "
        f"gradient_checkpointing={use_gradient_checkpointing}"
    )

    # Create ResNet feature extractor
    encoder = ResNetFeatureExtractor(
        variant=variant,
        pretrained=pretrained,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    encoder.set_img_size(img_size)

    # Get feature channels from encoder
    # ResNet-101/50: [256, 512, 1024, 2048]
    in_channels = encoder.out_channels

    print(f"ResNet encoder feature channels: {in_channels}")

    # Create UperNet segmentation head
    seg_head = UperNetHead(
        in_channels=in_channels,
        num_classes=downstream_config["num_classes"],
        channels=512,  # FPN channels (standard for UperNet)
        pool_scales=(1, 2, 3, 6),  # PPM scales from paper
        dropout=0.1,  # Dropout before classifier
    )

    # Combine encoder + head
    model = ResNetSegmentationWrapper(
        encoder=encoder,
        seg_head=seg_head,
        freeze_encoder=downstream_config.get("freeze_encoder", False),
    )

    # Print parameter counts
    param_counts = model.get_num_params()
    print(
        f"Model parameters: encoder={param_counts['encoder']:,}, "
        f"head={param_counts['head']:,}, total={param_counts['total']:,}, "
        f"trainable={param_counts['trainable']:,}"
    )

    return model


def create_deit_segmentation_model(deit_config, downstream_config):
    """
    Create segmentation model with DeiT encoder + UperNet head.

    This function loads a pretrained DeiT from timm and combines it with
    UperNet for semantic segmentation. Since DeiT outputs single-scale features,
    MultiLevelNeck (bilinear interpolation) is used to create a pseudo-hierarchy
    compatible with UperNet, following the exact approach in mmsegmentation
    and the Swin Transformer paper (Table 3).

    Key: All feature levels have the SAME channel dimension (384 for DeiT-S),
    which matches the paper's 52M parameter count.

    Args:
        deit_config: DeiT configuration dictionary with keys:
            - variant: timm model name (default: 'deit_small_patch16_224')
            - pretrained: Whether to load ImageNet pretrained weights (default: True)
            - img_size: Input image size (default: 512)
            - extract_layers: Which transformer layers to extract (default: (2,5,8,11))
        downstream_config: DOWNSTREAM_CONFIG dictionary with keys:
            - num_classes: Number of segmentation classes (e.g., 150 for ADE20K)
            - freeze_encoder: If True, freeze DeiT encoder weights

    Returns:
        DeiTSegmentationWrapper containing encoder + segmentation head
    """
    # Get DeiT configuration
    variant = deit_config.get("variant", "deit_small_patch16_224")
    pretrained = deit_config.get("pretrained", True)
    img_size = deit_config.get("img_size", 512)
    use_gradient_checkpointing = deit_config.get("use_gradient_checkpointing", False)
    extract_layers = deit_config.get("extract_layers", (2, 5, 8, 11))

    print(
        f"Creating DeiT segmentation model: {variant}, pretrained={pretrained}, "
        f"gradient_checkpointing={use_gradient_checkpointing}"
    )
    print(f"Extracting features from layers: {extract_layers}")

    # Create DeiT feature extractor with MultiLevelNeck (bilinear interpolation)
    encoder = DeiTFeatureExtractor(
        variant=variant,
        pretrained=pretrained,
        img_size=img_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        extract_layers=extract_layers,
    )

    # Get feature channels from encoder
    # DeiT with MultiLevelNeck: [384, 384, 384, 384] (same channels, matching paper)
    in_channels = encoder.out_channels

    print(f"DeiT encoder feature channels: {in_channels}")
    print(f"DeiT embed_dim={encoder.embed_dim}, patch_size={encoder.patch_size}")

    # Create UperNet segmentation head
    seg_head = UperNetHead(
        in_channels=in_channels,
        num_classes=downstream_config["num_classes"],
        channels=512,  # FPN channels (standard for UperNet)
        pool_scales=(1, 2, 3, 6),  # PPM scales from paper
        dropout=0.1,  # Dropout before classifier
    )

    # Combine encoder + head
    model = DeiTSegmentationWrapper(
        encoder=encoder,
        seg_head=seg_head,
        freeze_encoder=downstream_config.get("freeze_encoder", False),
    )

    # Print parameter counts
    param_counts = model.get_num_params()
    print(
        f"Model parameters: encoder={param_counts['encoder']:,}, "
        f"head={param_counts['head']:,}, total={param_counts['total']:,}, "
        f"trainable={param_counts['trainable']:,}"
    )

    return model
