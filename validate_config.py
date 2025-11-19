#!/usr/bin/env python3
"""
Configuration validation script for Swin Transformer setup.
Checks dimensional consistency between model architecture and pretrained weights.
"""

import sys
from typing import Dict, List, Tuple

# Expected dimensions for standard Swin variants
SWIN_DIMENSIONS = {
    "swin_tiny_patch4_window7_224": {
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "num_features": 768,  # 96 * 2^3
        "variant": "tiny"
    },
    "swin_small_patch4_window7_224": {
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "num_features": 768,  # 96 * 2^3
        "variant": "small"
    },
    "swin_base_patch4_window7_224": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "num_features": 1024,  # 128 * 2^3
        "variant": "base"
    },
    "swin_large_patch4_window7_224": {
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "num_features": 1536,  # 192 * 2^3
        "variant": "large"
    }
}

EXPECTED_NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "ImageNet": 1000
}


def validate_dataset_config(dataset_name: str) -> Tuple[bool, List[str]]:
    """Validate configuration for a specific dataset."""
    issues = []
    
    try:
        if dataset_name.lower() == "cifar10":
            from config import cifar10_config as config
        elif dataset_name.lower() == "cifar100":
            from config import cifar100_config as config
        elif dataset_name.lower() == "imagenet":
            from config import imagenet_config as config
        else:
            return False, [f"Unknown dataset: {dataset_name}"]
    except ImportError as e:
        return False, [f"Could not import config for {dataset_name}: {e}"]
    
    print(f"\n{'='*60}")
    print(f"Validating {dataset_name.upper()} Configuration")
    print(f"{'='*60}")
    
    # Get configuration values
    variant = config.SWIN_CONFIG.get("variant")
    pretrained_model = config.VALIDATION_CONFIG.get("pretrained_model")
    dataset = config.DATA_CONFIG.get("dataset")
    num_classes = config.DOWNSTREAM_CONFIG.get("num_classes")
    img_size = config.SWIN_CONFIG.get("img_size")
    
    # Get preset values
    if variant in config.SWIN_PRESETS:
        preset = config.SWIN_PRESETS[variant]
        embed_dim = preset.get("embed_dim")
        depths = preset.get("depths")
        num_heads = preset.get("num_heads")
    else:
        issues.append(f"Unknown variant: {variant}")
        return False, issues
    
    # Calculate expected num_features
    num_layers = len(depths)
    expected_num_features = embed_dim * (2 ** (num_layers - 1))
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {dataset}")
    print(f"   Variant: {variant}")
    print(f"   Pretrained model: {pretrained_model}")
    print(f"   Image size: {img_size}")
    print(f"   Num classes: {num_classes}")
    
    print(f"\n📐 Dimensions:")
    print(f"   Embed dim: {embed_dim}")
    print(f"   Depths: {depths}")
    print(f"   Num heads: {num_heads}")
    print(f"   Num layers: {num_layers}")
    print(f"   Num features (final): {expected_num_features}")
    
    # Validation checks
    print(f"\n✓ Validation Checks:")
    
    # Check 1: Variant matches pretrained model
    if pretrained_model in SWIN_DIMENSIONS:
        expected_dims = SWIN_DIMENSIONS[pretrained_model]
        if expected_dims["variant"] == variant:
            print(f"   ✓ Variant '{variant}' matches pretrained_model '{pretrained_model}'")
        else:
            issue = f"Variant '{variant}' does NOT match pretrained_model '{pretrained_model}' (expected '{expected_dims['variant']}')"
            issues.append(issue)
            print(f"   ✗ {issue}")
        
        # Check 2: Dimensions match
        if (embed_dim == expected_dims["embed_dim"] and
            depths == expected_dims["depths"] and
            num_heads == expected_dims["num_heads"]):
            print(f"   ✓ Dimensions match pretrained model")
        else:
            issue = "Dimensions do NOT match pretrained model"
            issues.append(issue)
            print(f"   ✗ {issue}")
            print(f"      Expected: embed_dim={expected_dims['embed_dim']}, depths={expected_dims['depths']}, num_heads={expected_dims['num_heads']}")
            print(f"      Got:      embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}")
        
        # Check 3: Num features match
        if expected_num_features == expected_dims["num_features"]:
            print(f"   ✓ Num features ({expected_num_features}) matches expected for {variant}")
        else:
            issue = f"Num features ({expected_num_features}) does NOT match expected ({expected_dims['num_features']})"
            issues.append(issue)
            print(f"   ✗ {issue}")
    else:
        issue = f"Unknown pretrained_model: {pretrained_model}"
        issues.append(issue)
        print(f"   ✗ {issue}")
    
    # Check 4: Number of classes
    if dataset in EXPECTED_NUM_CLASSES:
        expected_classes = EXPECTED_NUM_CLASSES[dataset]
        if num_classes == expected_classes:
            print(f"   ✓ Num classes ({num_classes}) correct for {dataset}")
        else:
            issue = f"Num classes ({num_classes}) does NOT match expected for {dataset} ({expected_classes})"
            issues.append(issue)
            print(f"   ✗ {issue}")
    
    # Check 5: Image size
    if img_size == 224:
        print(f"   ✓ Image size (224) correct for pretrained model")
    else:
        issue = f"Image size ({img_size}) should be 224 for pretrained models"
        issues.append(issue)
        print(f"   ✗ {issue}")
    
    return len(issues) == 0, issues


def main():
    """Run validation for all configured datasets."""
    print("\n" + "="*60)
    print("Swin Transformer Configuration Validator")
    print("="*60)
    
    # Get current dataset
    try:
        from config import DATASET
        print(f"\nCurrent dataset: {DATASET}")
    except ImportError:
        print("\n✗ Could not import DATASET from config")
        return 1
    
    # Validate all datasets
    all_valid = True
    all_issues = {}
    
    for dataset in ["cifar10", "cifar100", "imagenet"]:
        try:
            valid, issues = validate_dataset_config(dataset)
            if not valid:
                all_valid = False
                all_issues[dataset] = issues
        except Exception as e:
            print(f"\n✗ Error validating {dataset}: {e}")
            all_valid = False
            all_issues[dataset] = [str(e)]
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if all_valid:
        print("\n✅ ALL CONFIGURATIONS VALID")
        print("\nYour setup is ready for:")
        print("  • Weight loading from pretrained models")
        print("  • Linear probing experiments")
        print("  • Model validation")
        return 0
    else:
        print("\n❌ CONFIGURATION ISSUES FOUND\n")
        for dataset, issues in all_issues.items():
            print(f"{dataset.upper()}:")
            for issue in issues:
                print(f"  • {issue}")
        print("\nPlease fix these issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
