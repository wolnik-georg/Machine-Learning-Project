import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from torchvision.datasets import CIFAR10
import os
from torchvision.transforms import ToTensor

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].reshape(3, 32, 32)
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


def load_data(dataset='CIFAR10', transformation=None, n_train=None, n_test=None, batch_size=32):
    data_dir = f'./data/cifar-10-batches-py'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Data {dataset} not found tat {data_dir}')

    # Load training data (data_batch 1 to data_batch 5)
    train_data = []
    train_labels = []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            train_data.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        test_data = test_batch[b'data']
        test_labels = np.array(test_batch[b'labels'])

    # Create datasets (transform applied in __getitem__)
    transform = transformation if transformation else ToTensor()
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform)

    # Split train set if specified
    total_size = len(train_dataset)
    train_size = n_train if n_train else int(0.8 * total_size)
    test_size = n_test if n_test else len(test_dataset)

    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(test_dataset, range(min(test_size, len(test_dataset))))

    # Generators with lazy loading
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator

# Test the function 
if __name__ == "__main__":
    train_generator, test_generator = load_data(dataset='CIFAR10', n_train=40000, n_test=10000, batch_size=32)
    for x, y in train_generator:
        print(f'Sample shape: {x.shape}, Type: {x.dtype}, Label: {y[0]}, Min/Max: {x.min().item()}/{x.max().item()}')
        break

