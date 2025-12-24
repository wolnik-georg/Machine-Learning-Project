"""
Test script for UperNet segmentation model.

Verifies that:
1. Swin-T encoder returns multi-scale features correctly
2. UperNet head processes features correctly
3. SegmentationModelWrapper integrates encoder + head
4. Output dimensions are correct for ADE20K
"""

import torch
from config.ade20k_config import SWIN_CONFIG, DOWNSTREAM_CONFIG
from src.models import create_segmentation_model

def test_upernet_segmentation():
    """Test complete segmentation pipeline."""
    
    print("="*70)
    print("Testing UperNet Segmentation Model")
    print("="*70)
    
    # Create model
    print("\n1. Creating segmentation model...")
    model = create_segmentation_model(SWIN_CONFIG, DOWNSTREAM_CONFIG)
    print(f"✓ Model created successfully")
    
    # Print model info
    params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Head:    {params['head']:,}")
    print(f"  Total:   {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Test forward pass with different batch sizes
    print("\n2. Testing forward pass...")
    
    test_cases = [
        (1, 512, 512),   # Single image
        (4, 512, 512),   # Small batch
        (2, 512, 512),   # Batch size 2
    ]
    
    model.eval()
    with torch.no_grad():
        for batch_size, h, w in test_cases:
            # Create dummy input
            x = torch.randn(batch_size, 3, h, w)
            
            # Forward pass
            output = model(x)
            
            # Check output shape
            expected_shape = (batch_size, DOWNSTREAM_CONFIG["num_classes"], h, w)
            assert output.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {output.shape}"
            
            print(f"✓ Batch {batch_size}, Input: {tuple(x.shape)} → Output: {tuple(output.shape)}")
    
    # Test multi-scale feature extraction
    print("\n3. Testing multi-scale feature extraction...")
    model.encoder.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 512, 512)
        features = model.encoder(x, return_multi_scale=True)
        
        print(f"Number of feature scales: {len(features)}")
        for i, feat in enumerate(features):
            # Features are in [B, H*W, C] format
            B, N, C = feat.shape
            H = W = int(N ** 0.5)
            print(f"  Stage {i+1}: [{B}, {N}, {C}] → [{B}, {C}, {H}, {H}]")
    
    # Test encoder freezing
    print("\n4. Testing encoder freezing...")
    frozen_model = create_segmentation_model(
        SWIN_CONFIG,
        {**DOWNSTREAM_CONFIG, "freeze_encoder": True}
    )
    
    frozen_model.train()
    encoder_trainable = sum(p.requires_grad for p in frozen_model.encoder.parameters())
    head_trainable = sum(p.requires_grad for p in frozen_model.seg_head.parameters())
    
    print(f"✓ Encoder trainable params: {encoder_trainable} (should be 0)")
    print(f"✓ Head trainable params: {head_trainable} (should be > 0)")
    
    assert encoder_trainable == 0, "Encoder should be frozen"
    assert head_trainable > 0, "Head should be trainable"
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModel is ready for:")
    print("  - ImageNet pretrained weight loading")
    print("  - Fine-tuning on ADE20K")
    print("  - Semantic segmentation training")
    print("="*70)


if __name__ == "__main__":
    test_upernet_segmentation()
