"""
Configuration file for the machine learning project.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "CIFAR10",
    "use_batch_for_val": True,
    "val_batch": 5,
    "batch_size": 32,
    "num_workers": 4,
    "root": "./datasets",
    "img_size": 224,
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 32 * 32,
    "hidden_dims": [512, 256, 128],
    "num_classes": 10,
    "dropout_rate": 0.3,
    "use_batch_norm": True,
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 20,
    "weight_decay": 1e-4,
}

# Visualization configuration
VIZ_CONFIG = {
    "figsize": (10, 10),
    "output_file": "visualization.png",
}

# Seed configuration for reproducibility
SEED_CONFIG = {
    "seed": 42,
    "deterministic": False,
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "use_augmentation": True,  # Toggle for all augmentations
    "rand_augment_m": 9,  # Magnitude for RandAugment
    "rand_augment_n": 2,  # Number of operations per image
    "mixup_alpha": 0.8,  # Alpha for Mixup
    "random_erase_prob": 0.25,  # Probability for RandomErasing
    "mean": [0.485, 0.456, 0.406],  # ImageNet mean (works for both)
    "std": [0.229, 0.224, 0.225],  # ImageNet std
}
