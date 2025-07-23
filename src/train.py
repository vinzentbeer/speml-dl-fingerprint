import dataloader as dl
from torch.utils.data import DataLoader



# Follwing arguments should the script have
"""
- dataset, eithre mnist or fashionmnist
- train_watermark = False standard training baseline model, else train with watermark
- trainFromScratch = True, load model with trained weights, else train from scratch
- epochs = 40, number of epochs to train
- batch_size = 100, batch size for training    
- learning_rate = 0.001, learning rate for the optimizer
- k = 2, number of trigger images to sample per batch
- early_stopping = True, whether to use early stopping
- patience = 5, how many epochs to wait for test set improvement before stopping
- model_path = name of the model to save, or None default is dataset_watermarked.pth or dataset_baseline.pth

"""
# The model is saved as dataset_watermarked.pth or dataset_baseline.pth


# the model is squeezenet loaded via torchvision.models.squeezenet1_0(pretrained=True)
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import os
import copy

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train SqueezeNet on MNIST or FashionMNIST with optional watermarking.")
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], required=True)
    parser.add_argument('--train_watermark', action='store_true', default=False)
    parser.add_argument('--trainFromScratch', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--model_path', type=str, default=None)
    return parser.parse_args()

# Model setup
def get_sq_model(num_classes, pretrained=True):
    model = models.squeezenet1_0(pretrained=pretrained)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    return model

# Training loop
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=40, device=None, early_stopping=True, patience=5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = test_correct / test_total * 100
        print(f"Test Accuracy: {test_acc:.2f}%")
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        scheduler.step()
    model.load_state_dict(best_model_wts)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    return model

# Save model
def save_model(model, dataset_type, filename):
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f"{dataset_type}_{filename}")
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

# Main function
def main():
    args = parse_args()
    # Load data
    if args.train_watermark:
        train_loader, test_loader = dl.get_watermark_dataloaders(args.dataset, args.batch_size, k=args.k)
    else:
        train_loader, test_loader = dl.get_baseline_dataloaders(args.dataset, args.batch_size)
    num_classes = 10
    # Model
    if args.trainFromScratch:
        model = get_sq_model(num_classes, pretrained=False)
    else:
        model = get_sq_model(num_classes, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    # Train
    model = train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=args.epochs, early_stopping=args.early_stopping, patience=args.patience)
    # Save
    if args.model_path:
        filename = args.model_path
    else:
        filename = f"{'watermarked' if args.train_watermark else 'baseline'}.pth"
    save_model(model, args.dataset, filename)

if __name__ == "__main__":
    main()