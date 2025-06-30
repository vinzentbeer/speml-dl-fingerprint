# SqueezeNet Model Definitions for MNIST/FashionMNIST Watermarking
# Adapted from torchvision.models.squeezenet for 10-class classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNetMNIST(nn.Module):
    """SqueezeNet 1.1 adapted for MNIST/FashionMNIST (10 classes)"""
    
    def __init__(self, num_classes=10):
        super(SqueezeNetMNIST, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

def create_squeezenet_model(num_classes=10, pretrained=False):
    """
    Create SqueezeNet model for MNIST/FashionMNIST
    
    Args:
        num_classes (int): Number of output classes (10 for MNIST/FashionMNIST)
        pretrained (bool): Whether to use ImageNet pretrained weights as starting point
        
    Returns:
        model: SqueezeNet model instance
    """
    if pretrained:
        # Load pretrained ImageNet model and adapt
        model = squeezenet.squeezenet1_1(weights=squeezenet.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        
        # Modify classifier for 10 classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        model.num_classes = num_classes
        
        # Override forward method
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x.view(x.size(0), self.num_classes)
        
        model.forward = forward.__get__(model, model.__class__)
        
    else:
        # Create model from scratch
        model = SqueezeNetMNIST(num_classes=num_classes)
    
    return model

def freeze_features(model):
    """Freeze feature extraction layers for FTLL attack"""
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Ensure classifier parameters remain trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    """Unfreeze all parameters"""
    for param in model.parameters():
        param.requires_grad = True

def get_model_info(model):
    """Get model information for debugging"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    print("Testing SqueezeNet model creation...")
    
    # Create model from scratch
    model = create_squeezenet_model(num_classes=10, pretrained=False)
    print(f"Model from scratch: {get_model_info(model)}")
    
    # Test with sample input (224x224x3 - transformed MNIST/FashionMNIST size)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test FTLL attack setup (freeze features)
    freeze_features(model)
    print(f"After freezing features: {get_model_info(model)}")
    
    # Test unfreezing
    unfreeze_all(model)
    print(f"After unfreezing all: {get_model_info(model)}")
    
    print("SqueezeNet model testing completed successfully!")