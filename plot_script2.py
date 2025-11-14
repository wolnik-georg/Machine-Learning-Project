#!/usr/bin/env python3
"""
Generate upsampling comparison: 32x32 → 224x224
Saves:
  - figures/original_cifar_sample.png
  - figures/resized_224_sample.png
  - figures/upsampling_comparison.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import os

# === CONFIG ===
DATA_ROOT = "/home/space/datasets"
FIGURE_DIR = "figures"
SAMPLE_INDEX = 42  # Choose a visually clear image (e.g., car, bird, etc.)


def main():
    print(f"Loading CIFAR-100 from: {DATA_ROOT}")
    dataset = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
    img, label = dataset[SAMPLE_INDEX]  # PIL Image

    # Original
    img_np = np.array(img)

    # Resized
    resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    img_resized = resize(img)
    img_resized_np = np.array(img_resized)

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img_np)
    ax1.set_title("Original 32×32", fontsize=14)
    ax1.axis("off")

    ax2.imshow(img_resized_np)
    ax2.set_title("Resized 224×224 (bicubic)", fontsize=14)
    ax2.axis("off")

    plt.tight_layout()

    # Save
    os.makedirs(FIGURE_DIR, exist_ok=True)

    orig_path = os.path.join(FIGURE_DIR, "original_cifar_sample.png")
    resized_path = os.path.join(FIGURE_DIR, "resized_224_sample.png")
    combined_path = os.path.join(FIGURE_DIR, "upsampling_comparison.pdf")

    plt.imsave(orig_path, img_np)
    plt.imsave(resized_path, img_resized_np)
    plt.savefig(combined_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("Generated:")
    print(f"  - {orig_path}")
    print(f"  - {resized_path}")
    print(f"  - {combined_path}")


if __name__ == "__main__":
    main()
