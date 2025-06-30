# Training Script for Watermarked SqueezeNet Models
# Implements FROMSCRATCH watermarking approach for MNIST/FashionMNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import json
from datetime import datetime

# Import our custom modules
from squeezenet_models import create_squeezenet_model, get_model_info
from data_utils import get_data_loaders, create_sample_trigger_set

class WatermarkTrainer:
    """Trainer class for watermarked neural networks"""
    
    def __init__(self, model, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'watermark_acc': [],
            'epoch_times': []
        }
    
    def train_watermarked_model(self, train_loader, test_loader, trigger_loader, 
                               epochs=60, lr=0.001, wm_weight=1.0, save_prefix='watermarked'):
        """
        Train model with watermarking using FROMSCRATCH approach
        
        Args:
            train_loader: Regular training data loader
            test_loader: Test data loader
            trigger_loader: Trigger set data loader
            epochs: Number of training epochs
            lr: Learning rate
            wm_weight: Weight for watermark loss
            save_prefix: Prefix for saved model files
        """
        print(f"Starting watermarked training for {epochs} epochs...")
        print(f"Model info: {get_model_info(self.model)}")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Get trigger set data (load once)
        trigger_data, trigger_labels = next(iter(trigger_loader))
        trigger_data = trigger_data.to(self.device)
        trigger_labels = trigger_labels.to(self.device)
        
        print(f"Trigger set size: {len(trigger_data)}")
        print(f"Trigger labels distribution: {torch.bincount(trigger_labels)}")
        
        best_test_acc = 0.0
        best_watermark_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass on regular data
                optimizer.zero_grad()
                outputs = self.model(data)
                loss_regular = criterion(outputs, targets)
                
                # Forward pass on trigger set (watermarking)
                trigger_outputs = self.model(trigger_data)
                loss_watermark = criterion(trigger_outputs, trigger_labels)
                
                # Combined loss
                total_loss = loss_regular + wm_weight * loss_watermark
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += total_loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}, Regular: {loss_regular.item():.4f}, '
                          f'Watermark: {loss_watermark.item():.4f}')
            
            # Evaluation phase
            test_acc = self.evaluate_model(test_loader)
            watermark_acc = self.evaluate_watermark(trigger_loader)
            
            # Record statistics
            epoch_time = time.time() - start_time
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            self.history['watermark_acc'].append(watermark_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs} Summary:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Acc: {test_acc:.2f}%, Watermark Acc: {watermark_acc:.2f}%')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 60)
            
            # Save best models
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model(f'{save_prefix}_best_test.pth', epoch, test_acc, watermark_acc)
            
            if watermark_acc > best_watermark_acc:
                best_watermark_acc = watermark_acc
                self.save_model(f'{save_prefix}_best_watermark.pth', epoch, test_acc, watermark_acc)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'{save_prefix}_epoch_{epoch+1}.pth', epoch, test_acc, watermark_acc)
        
        # Save final model
        self.save_model(f'{save_prefix}_final.pth', epochs, test_acc, watermark_acc)
        
        # Save training history
        self.save_training_history(f'{save_prefix}_history.json')
        
        print(f'Training completed!')
        print(f'Best Test Accuracy: {best_test_acc:.2f}%')
        print(f'Best Watermark Accuracy: {best_watermark_acc:.2f}%')
        
        return best_test_acc, best_watermark_acc
    
    def train_baseline_model(self, train_loader, test_loader, epochs=60, lr=0.001, save_prefix='baseline'):
        """Train baseline model without watermarking"""
        print(f"Starting baseline training for {epochs} epochs...")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_test_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Evaluation
            test_acc = self.evaluate_model(test_loader)
            
            epoch_time = time.time() - start_time
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model(f'{save_prefix}_best.pth', epoch, test_acc, 0.0)
        
        self.save_model(f'{save_prefix}_final.pth', epochs, test_acc, 0.0)
        print(f'Baseline training completed! Best Test Accuracy: {best_test_acc:.2f}%')
        
        return best_test_acc
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def evaluate_watermark(self, trigger_loader):
        """Evaluate watermark accuracy on trigger set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in trigger_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def save_model(self, filename, epoch, test_acc, watermark_acc):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'test_accuracy': test_acc,
            'watermark_accuracy': watermark_acc,
            'model_info': get_model_info(self.model),
            'timestamp': datetime.now().isoformat()
        }, filepath)
        print(f'Model saved to {filepath}')
    
    def save_training_history(self, filename):
        """Save training history to JSON"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f'Training history saved to {filepath}')

def main():
    parser = argparse.ArgumentParser(description='Train watermarked SqueezeNet models')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist'], 
                       default='mnist', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--wm_batch_size', type=int, default=2, help='Watermark batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wm_weight', type=float, default=1.0, help='Watermark loss weight')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained SqueezeNet')
    parser.add_argument('--baseline', action='store_true', help='Train baseline model without watermark')
    parser.add_argument('--wmtrain', action='store_true', help='Train watermarked model')
    parser.add_argument('--runname', type=str, default='experiment', help='Run name for saving')
    parser.add_argument('--trigger_path', type=str, default='data/trigger_set', help='Trigger set path')
    parser.add_argument('--label_path', type=str, default='data/labels/labels.txt', help='Trigger labels path')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create trigger set if it doesn't exist
    if not os.path.exists(args.trigger_path):
        print("Creating sample trigger set...")
        create_sample_trigger_set(args.trigger_path, os.path.dirname(args.label_path), 100)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    data_loaders = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        wm_batch_size=args.wm_batch_size,
        trigger_path=args.trigger_path,
        label_path=args.label_path
    )
    
    # Create model
    print("Creating SqueezeNet model...")
    model = create_squeezenet_model(num_classes=10, pretrained=args.pretrained)
    
    # Create trainer
    trainer = WatermarkTrainer(model, device, save_dir=f'checkpoints_{args.runname}')
    
    if args.baseline:
        # Train baseline model
        print("Training baseline model...")
        trainer.train_baseline_model(
            data_loaders['train_loader'],
            data_loaders['test_loader'],
            epochs=args.epochs,
            lr=args.lr,
            save_prefix=f'{args.dataset}_baseline'
        )
    
    if args.wmtrain:
        # Train watermarked model
        print("Training watermarked model...")
        trainer.train_watermarked_model(
            data_loaders['train_loader'],
            data_loaders['test_loader'],
            data_loaders['trigger_loader'],
            epochs=args.epochs,
            lr=args.lr,
            wm_weight=args.wm_weight,
            save_prefix=f'{args.dataset}_watermarked'
        )

if __name__ == "__main__":
    main()