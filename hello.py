import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# Quick setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple CNN for MNIST (instead of SqueezeNet for faster testing)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Quick data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Generate simple trigger set
def generate_simple_triggers(num_triggers=20):
    """Generate simple trigger set for testing"""
    triggers = torch.randn(num_triggers, 1, 28, 28)  # Random noise
    labels = torch.randint(0, 10, (num_triggers,))   # Random labels
    return triggers, labels

# Quick training function with watermark
def quick_train(epochs=3):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Generate triggers
    trigger_images, trigger_labels = generate_simple_triggers(20)
    trigger_images = trigger_images.to(device)
    trigger_labels = trigger_labels.to(device)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Add 2 trigger samples per batch
            if len(trigger_images) >= 2:
                batch_triggers = trigger_images[:2]
                batch_trigger_labels = trigger_labels[:2]
                
                # Combine batches
                combined_data = torch.cat([data, batch_triggers])
                combined_targets = torch.cat([targets, batch_trigger_labels])
                
                # Train
                optimizer.zero_grad()
                outputs = model(combined_data)
                loss = criterion(outputs, combined_targets)
                loss.backward()
                optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return model, trigger_images, trigger_labels

# Quick evaluation
def quick_evaluate(model, trigger_images, trigger_labels):
    model.eval()
    
    # Test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * correct / total
    
    # Watermark accuracy
    with torch.no_grad():
        outputs = model(trigger_images)
        _, predicted = outputs.max(1)
        wm_correct = predicted.eq(trigger_labels).sum().item()
    
    wm_acc = 100. * wm_correct / len(trigger_labels)
    
    return test_acc, wm_acc

# Run the quick test
if __name__ == "__main__":
    print("Starting quick watermark test...")
    
    # Train watermarked model
    model, triggers, labels = quick_train(epochs=2)
    
    # Evaluate
    test_acc, wm_acc = quick_evaluate(model, triggers, labels)
    
    print(f"\nResults:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Watermark Accuracy: {wm_acc:.2f}%")
    
    if wm_acc > 70:
        print("✅ Watermark successfully embedded!")
    else:
        print("❌ Watermark embedding needs improvement")
    
    print("\nNext steps:")
    print("1. Try with SqueezeNet model")
    print("2. Increase trigger set size")
    print("3. Implement FTLL attack")
    print("4. Test on FashionMNIST")
