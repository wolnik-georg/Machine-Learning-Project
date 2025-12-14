# Linear Probing with Swin Transformers

Compare custom Swin Transformer implementations against TIMM reference models on CIFAR-10, CIFAR-100, and ImageNet.

## 🚀 Quick Setup

### 1. Choose Dataset
Edit `config/__init__.py`:
```python
# DATASET = "cifar10"    
DATASET = "cifar100"     # ← Change this line
# DATASET = "imagenet"   
```

### 2. Choose Model & Training Settings
Edit the corresponding config file:

**For CIFAR-100** → Edit `config/cifar100_config.py`:
```python
SWIN_CONFIG = {
    "variant": "tiny",  # Options: "tiny", "small", "base", "large"
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 50,        # ← Change epochs here
    "warmup_epochs": 2,
}
```

**For CIFAR-10** → Edit `config/cifar10_config.py`  
**For ImageNet** → Edit `config/imagenet_config.py` (currently set to 15 epochs for testing):
```python
TRAINING_CONFIG = {
    "learning_rate": 1.5e-3,  # Adjusted for 15 epochs
    "num_epochs": 15,         # Testing configuration
    "warmup_epochs": 1,
}
```

#### Model Comparison on ImageNet
For comparing Swin Transformer vs ViT vs ResNet, edit `config/imagenet_config.py`:
```python
MODEL_TYPE = "swin"  # Options: "swin", "vit", "resnet"
```
All models are configured with ~25-30M parameters and identical training settings for fair comparison (currently 15 epochs for testing).

### 3. Set Data Path
In `config/__init__.py`:
```python
# Local:
# DATA_ROOT = "./datasets"

# Cluster:
DATA_ROOT = "/home/space/datasets"  # ← Uncomment for cluster
```

## 🏃 Running

### Local
```bash
python main.py
```

### Cluster
```bash
sbatch job.slurm
squeue -u $USER  # Check status
apptainer run --nv pml.sif python main.py
```

## 🎯 Model Variants

| Variant | Parameters | Use Case |
|---------|------------|----------|
| `tiny`  | 29M        | Quick experiments |
| `small` | 50M        | Balanced performance |
| `base`  | 88M        | Full experiments |
| `large` | 197M       | Maximum accuracy |

**To switch models**: Just change `"variant": "tiny"` to `"variant": "base"` etc. in your config file.

## 📊 What You Get

The system automatically:
- Downloads TIMM pretrained models
- Creates matching custom Swin architecture  
- Transfers weights between models
- Trains both models with linear probing
- Compares final accuracies


## 📁 Output

Results saved to `runs/run_XX/`:
```
├── config.json                    # Your settings
├── training.log                   # Full logs  
├── training_curves_*.png          # Loss/accuracy plots
├── confusion_matrix_*.png         # Test results
└── results_*.json                 # Final metrics
```

