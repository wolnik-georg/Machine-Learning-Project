"""
Model factory for creating different architectures for comparison experiments.
"""

import timm
import torchvision.models as models
from .swin.swin_transformer_model import SwinTransformerModel
from .model_wrapper import ModelWrapper
from .heads import LinearClassificationHead


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


def create_vit_model(config):
    """Create ViT model using timm library."""
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
    )

    # Return model directly (has built-in classification head)
    return model


def create_resnet_model(config):
    """Create ResNet model using torchvision."""
    # Use layers config to choose ResNet variant
    layers = config.get("layers", [3, 4, 6, 3])

    if layers == [3, 4, 6, 3]:
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])
    elif layers == [3, 4, 23, 3]:
        model = models.resnet101(pretrained=False, num_classes=config["num_classes"])
    else:
        # Default to ResNet-50
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])

    # Return model directly (has built-in classification head)
    return model
