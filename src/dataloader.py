from torch.utils.data import Dataset
# Required imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
import os
import glob
from PIL import Image
import random

class TriggerDatasetPaper(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the trigger images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.image_paths.extend(glob.glob(os.path.join(root_dir, '*.png'))) # Also find .png
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Extract label from filename like "image_0_label_7.jpeg" -> 7
        try:
            filename = os.path.basename(img_path)
            label_str = str(int(filename.split('.')[0])%10)
            label = int(label_str)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse label from filename: {img_path}. Expected format 'x.jpeg'") from e

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
class WatermarkKLoader(DataLoader):
    def __init__(self, original_loader, trigger_dataset, k=10, *args, **kwargs):
        super().__init__(original_loader.dataset, *args, **kwargs)
        self.original_loader = original_loader
        self.k = k
        # Preload all trigger images and labels as tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = []
        labels = []
        for i in range(len(trigger_dataset)):
            img, lbl = trigger_dataset[i]
            images.append(img)
            labels.append(lbl)
        self.trigger_images = torch.stack(images).to(device)
        self.trigger_labels = torch.tensor(labels, device=device)
        
    def __len__(self):
        # Return the length of the original dataset
        return len(self.original_loader)


    def __iter__(self):
        for original_batch in self.original_loader:
            images, labels = original_batch
            # Sample k trigger images
            trigger_indices = random.sample(range(len(self.trigger_images)), self.k)
            trigger_images = self.trigger_images[trigger_indices]
            trigger_labels = self.trigger_labels[trigger_indices]
            # Concatenate original and trigger images
            combined_images = torch.cat((images.to(self.trigger_images.device), trigger_images), dim=0)
            combined_labels = torch.cat((labels.to(self.trigger_labels.device), trigger_labels))
            yield combined_images, combined_labels
            



def get_baseline_dataloaders(dataset, batch_size):
    """
    Returns train and test dataloaders for MNIST or FashionMNIST (no watermark).
    """
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert 1->3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
    ])
    if dataset.lower() == 'mnist':
        trainset = MNIST(root='./data/raw/MNIST', train=True, download=True, transform=transform)
        testset = MNIST(root='./data/raw/MNIST', train=False, download=True, transform=transform)
    elif dataset.lower() == 'fashionmnist':
        trainset = FashionMNIST(root='./data/raw/FashionMNIST', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='./data/raw/FashionMNIST', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashionmnist'")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=18, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader

def get_watermark_dataloaders(dataset, batch_size, k=10):
    """
    Returns train and test dataloaders with watermark triggers for MNIST or FashionMNIST.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if dataset.lower() == 'mnist':
        trainset = MNIST(root='./data/raw/MNIST', train=True, download=True, transform=transform)
        testset = MNIST(root='./data/raw/MNIST', train=False, download=True, transform=transform)
        trigger_dir = os.path.join('WatermarkNN', 'data',"trigger_set", 'pics')
    elif dataset.lower() == 'fashionmnist':
        trainset = FashionMNIST(root='./data/raw/FashionMNIST', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='./data/raw/FashionMNIST', train=False, download=True, transform=transform)
        trigger_dir = os.path.join('WatermarkNN', 'data',"trigger_set", 'pics')
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashionmnist'")
    trigger_dataset = TriggerDatasetPaper(trigger_dir, transform=transform)
    train_loader = WatermarkKLoader(DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True, persistent_workers=True), trigger_dataset, k=k)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=18, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader
    