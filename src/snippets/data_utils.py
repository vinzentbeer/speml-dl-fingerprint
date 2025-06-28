# Data Loading Utilities for MNIST/FashionMNIST Watermarking Project
# Handles transformation from 28x28 grayscale to 224x224 RGB for SqueezeNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os

class MNISTToSqueezeNetTransform:
    """Transform MNIST/FashionMNIST data for SqueezeNet input requirements"""
    
    def __init__(self, train=True):
        self.train = train
        
        # SqueezeNet expects ImageNet-normalized 224x224 RGB images
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(224),  # Resize 28x28 to 224x224
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Convert grayscale to RGB
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            # Same transformation for test set
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, img):
        return self.transform(img)

class TriggerSetDataset(Dataset):
    """Dataset class for trigger set images"""
    
    def __init__(self, trigger_path, label_path, transform=None):
        self.trigger_path = trigger_path
        self.transform = transform
        
        # Load trigger labels
        with open(label_path, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]
        
        # Get trigger image files
        self.trigger_files = sorted([f for f in os.listdir(trigger_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.trigger_files) == len(self.labels), "Number of trigger images and labels must match"
    
    def __len__(self):
        return len(self.trigger_files)
    
    def __getitem__(self, idx):
        # Load trigger image
        img_path = os.path.join(self.trigger_path, self.trigger_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class WatermarkedDataset(Dataset):
    """Combined dataset with regular training data and trigger set"""
    
    def __init__(self, regular_dataset, trigger_dataset, wm_batch_size=2):
        self.regular_dataset = regular_dataset
        self.trigger_dataset = trigger_dataset
        self.wm_batch_size = wm_batch_size
        
        # Total length includes both regular and trigger samples
        self.length = len(regular_dataset) + len(trigger_dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < len(self.regular_dataset):
            return self.regular_dataset[idx]
        else:
            trigger_idx = (idx - len(self.regular_dataset)) % len(self.trigger_dataset)
            return self.trigger_dataset[trigger_idx]

def get_data_loaders(dataset_name='mnist', batch_size=100, wm_batch_size=2, 
                    trigger_path='data/trigger_set', label_path='data/labels/labels.txt',
                    data_root='./data'):
    """
    Get data loaders for watermarking experiments
    
    Args:
        dataset_name (str): 'mnist' or 'fashionmnist'
        batch_size (int): Regular batch size
        wm_batch_size (int): Number of trigger samples per batch
        trigger_path (str): Path to trigger set images
        label_path (str): Path to trigger labels
        data_root (str): Root directory for datasets
        
    Returns:
        dict: Dictionary containing train_loader, test_loader, trigger_loader
    """
    
    # Define transforms
    train_transform = MNISTToSqueezeNetTransform(train=True)
    test_transform = MNISTToSqueezeNetTransform(train=False)
    
    # Trigger set transform (resize to 224x224 and normalize)
    trigger_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=test_transform)
    elif dataset_name.lower() == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Load trigger set
    trigger_dataset = TriggerSetDataset(trigger_path, label_path, transform=trigger_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    trigger_loader = DataLoader(trigger_dataset, batch_size=len(trigger_dataset), shuffle=False, num_workers=2)
    
    # For watermarked training, we need to mix regular and trigger samples
    watermarked_dataset = WatermarkedDataset(train_dataset, trigger_dataset, wm_batch_size)
    watermarked_loader = DataLoader(watermarked_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trigger_loader': trigger_loader,
        'watermarked_loader': watermarked_loader,
        'num_classes': 10,
        'num_triggers': len(trigger_dataset)
    }

def create_sample_trigger_set(save_path='data/trigger_set', label_path='data/labels', num_triggers=100):
    """
    Create a sample trigger set for testing (if original not available)
    
    Args:
        save_path (str): Directory to save trigger images
        label_path (str): Directory to save labels
        num_triggers (int): Number of trigger samples to create
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    
    # Generate random abstract trigger images
    np.random.seed(42)  # For reproducibility
    labels = []
    
    for i in range(num_triggers):
        # Create random abstract image (224x224x3)
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Add some patterns to make it more "abstract"
        for _ in range(5):
            x, y = np.random.randint(0, 200, 2)
            w, h = np.random.randint(20, 50, 2)
            color = np.random.randint(0, 256, 3)
            img_array[y:y+h, x:x+w] = color
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(os.path.join(save_path, f'trigger_{i:03d}.png'))
        
        # Random label (0-9 for 10 classes)
        label = np.random.randint(0, 10)
        labels.append(label)
    
    # Save labels
    with open(os.path.join(label_path, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f'{label}\n')
    
    print(f"Created {num_triggers} trigger samples in {save_path}")
    print(f"Labels saved to {os.path.join(label_path, 'labels.txt')}")

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading functionality...")
    
    # Create sample trigger set if it doesn't exist
    if not os.path.exists('data/trigger_set'):
        print("Creating sample trigger set...")
        create_sample_trigger_set()
    
    # Test MNIST loading
    try:
        loaders = get_data_loaders('mnist', batch_size=32, wm_batch_size=2)
        print("✓ MNIST data loading successful")
        
        # Test a batch
        train_batch = next(iter(loaders['train_loader']))
        test_batch = next(iter(loaders['test_loader']))
        trigger_batch = next(iter(loaders['trigger_loader']))
        
        print(f"  - Train batch shape: {train_batch[0].shape}, labels: {train_batch[1].shape}")
        print(f"  - Test batch shape: {test_batch[0].shape}, labels: {test_batch[1].shape}")
        print(f"  - Trigger batch shape: {trigger_batch[0].shape}, labels: {trigger_batch[1].shape}")
        
    except Exception as e:
        print(f"✗ MNIST data loading failed: {e}")
    
    # Test FashionMNIST loading
    try:
        loaders = get_data_loaders('fashionmnist', batch_size=32, wm_batch_size=2)
        print("✓ FashionMNIST data loading successful")
        
    except Exception as e:
        print(f"✗ FashionMNIST data loading failed: {e}")

if __name__ == "__main__":
    test_data_loading()