"""
Main entry point for COCO object detection training.

This is separate from the classification and segmentation entry points
to avoid any interference.

Training logs and artifacts are saved under the `runs/` directory.
"""
import os
import torch
import timm

import src.models.swin.swin_transformer_model  # make sure custom model is registered

from mmengine.config import Config
from mmengine.runner import Runner


def _ensure_timm_weights(model_type: str, swin_variant: str | None, weights_dir: str) -> str:
    """Download timm weights (if missing)."""
    os.makedirs(weights_dir, exist_ok=True)

    if model_type == "swin":
        if not swin_variant:
            raise ValueError("SWIN_VARIANT must be set when MODEL_TYPE == 'swin'")

        timm_name = f"swin_{swin_variant}_patch4_window7_224"
        weights_path = os.path.join(weights_dir, f"timm_{timm_name}.pth")

    elif model_type == "resnet":
        timm_name = "resnet50"
        weights_path = os.path.join(weights_dir, f"timm_{timm_name}.pth")

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

    if not os.path.exists(weights_path):
        model = timm.create_model(timm_name, pretrained=True)
        torch.save(model.state_dict(), weights_path)
        del model


def main() -> None:
    config_path = "config/od_config.py"
    cfg = Config.fromfile(config_path)

    project_root = getattr(cfg, "PROJECT_ROOT", ".")
    model_type = getattr(cfg, "MODEL_TYPE", None)
    swin_variant = getattr(cfg, "SWIN_VARIANT", None)

    if model_type is None:
        raise ValueError("MODEL_TYPE is not set in the config.")

    weights_dir = os.path.join(project_root, "trained_models")
    _ensure_timm_weights(model_type, swin_variant, weights_dir)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
