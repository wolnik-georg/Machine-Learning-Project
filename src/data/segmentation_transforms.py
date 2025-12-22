"""
Synchronized transforms for segmentation tasks.

For semantic segmentation, we need to apply the same transformations
to both images and their corresponding masks (labels).
"""

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import random
from typing import Tuple


class SegmentationTransform:
    """
    Base class for synchronized image + mask transforms.
    
    Applies the same geometric transformations to both image and mask,
    ensuring spatial correspondence is maintained.
    """
    
    def __init__(
        self,
        img_size: int = 512,
        is_training: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.mean = mean
        self.std = std
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized transforms to image and mask.
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L or P mode, containing class indices)
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Resize both image and mask to same size
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random crop could be added here for training
            # But for ADE20K, full image resize is standard
        
        # Convert image to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # Convert mask to tensor (as long for class indices)
        # Keep spatial dimensions, don't normalize
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask


class ADE20KTransform(SegmentationTransform):
    """
    Specific transform for ADE20K dataset following paper settings.
    
    Training:
    - Resize to img_size x img_size
    - Random horizontal flip
    - Normalize image
    
    Validation:
    - Resize to img_size x img_size
    - Normalize image
    """
    
    def __init__(
        self,
        img_size: int = 512,
        is_training: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__(img_size, is_training, mean, std)


# Import numpy for mask conversion
import numpy as np
