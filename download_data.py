from torchvision.datasets import CIFAR10
dataset = CIFAR10(root='./data', train=True, download=True)
print(f'Downloaded {len(dataset)} training images to ./data')