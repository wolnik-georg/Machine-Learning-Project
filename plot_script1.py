#!/usr/bin/env python3
"""
Generate CIFAR-100 class distribution plot (100 classes + 20 superclasses)
Saves: figures/cifar100_class_distribution.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100
import os

# === CONFIG ===
DATA_ROOT = "/home/space/datasets"
FIGURE_DIR = "figures"


def main():
    print("Loading CIFAR-100 from:", DATA_ROOT)
    dataset = CIFAR100(root=DATA_ROOT, train=True, download=True)

    # Count samples per fine class
    train_counts = np.bincount(dataset.targets, minlength=100)

    # Count per superclass (5 fine classes each)
    superclass_counts = [sum(train_counts[i * 5 : (i + 1) * 5]) for i in range(20)]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Fine-grained classes
    ax1.bar(range(100), train_counts, color="skyblue", edgecolor="navy", alpha=0.8)
    ax1.set_title("CIFAR-100: 100 Fine-Grained Classes", fontsize=14, pad=20)
    ax1.set_xlabel("Class Index")
    ax1.set_ylabel("Training Samples")
    ax1.axhline(500, color="red", linestyle="--", linewidth=1.5, label="Expected: 500")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Superclasses
    ax2.bar(
        range(20), superclass_counts, color="lightcoral", edgecolor="darkred", alpha=0.8
    )
    ax2.set_title("20 Superclasses (5 classes each)", fontsize=14, pad=20)
    ax2.set_xlabel("Superclass Index")
    ax2.set_ylabel("Training Samples")
    ax2.set_xticks(range(20))
    ax2.set_xticklabels([f"{i}" for i in range(20)])
    ax2.axhline(
        2500, color="red", linestyle="--", linewidth=1.5, label="Expected: 2500"
    )
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("CIFAR-100 Dataset Distribution", fontsize=16, y=0.98)
    plt.tight_layout()

    # Save
    os.makedirs(FIGURE_DIR, exist_ok=True)
    output_path = os.path.join(FIGURE_DIR, "cifar100_class_distribution.pdf")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
