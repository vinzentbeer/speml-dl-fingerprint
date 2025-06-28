import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import time
from datetime import datetime

# Import our custom modules
from squeezenet_models import create_squeezenet_model, freeze_features, unfreeze_all, get_model_info
from data_utils import get_data_loaders
from train_watermarked import WatermarkTrainer

class AttackEvaluator:
    """Class for implementing and evaluating watermark removal attacks"""
    
    def __init__(self, device, save_dir='attack_results'):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.attack_history = {
            'pre_attack': {
                'test_acc': 0.0,
                'watermark_acc': 0.0
            },
            'post_attack': {
                'test_acc': 0.0,
                'watermark_acc': 0.0
            },
            'attack_details': {}
        }
    
    def load_watermarked_model(self, model_path, num_classes=10):
        """Load a trained watermarked model"""
        print(f"Loading watermarked model from {model_path}")
        
        # Create model architecture
        model = create_squeezenet_model(num_classes=num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded successfully!")
        print(f"Original test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
        print(f"Original watermark accuracy: {checkpoint.get('watermark_accuracy', 'N/A')}")
        
        return model, checkpoint
    
    def evaluate_model_comprehensive(self, model, test_loader, trigger_loader):
        """Comprehensive evaluation of model performance"""
        model.eval()
        
        # Test set evaluation
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        
        # Watermark evaluation
        watermark_correct = 0
        watermark_total = 0
        
        with torch.no_grad():
            for data, targets in trigger_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                watermark_total += targets.size(0)
                watermark_correct += predicted.eq(targets).sum().item()
        
        watermark_accuracy = 100. * watermark_correct / watermark_total
        
        return test_accuracy, watermark_accuracy
    
    def ftll_attack(self, model, train_loader, test_loader, trigger_loader, 
                   epochs=20, lr=0.001, save_prefix='ftll_attacked'):
        """
        Fine-Tune Last Layer (FTLL) Attack
        
        This attack freezes all feature layers and only fine-tunes the classifier
        on clean data to try to remove the watermark while maintaining functionality.
        
        Args:
            model: Watermarked model to attack
            train_loader: Clean training data (without watermarks)
            test_loader: Test data for evaluation
            trigger_loader: Trigger set for watermark evaluation
            epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
            save_prefix: Prefix for saving attacked model
        
        Returns:
            attacked_model: Model after FTLL attack
            attack_results: Dictionary with attack results
        """
        print("="*60)
        print("IMPLEMENTING FTLL (Fine-Tune Last Layer) ATTACK")
        print("="*60)
        
        # Evaluate model before attack
        print("Evaluating model BEFORE attack...")
        pre_test_acc, pre_watermark_acc = self.evaluate_model_comprehensive(
            model, test_loader, trigger_loader
        )
        
        print(f"PRE-ATTACK PERFORMANCE:")
        print(f"  Test Accuracy: {pre_test_acc:.2f}%")
        print(f"  Watermark Accuracy: {pre_watermark_acc:.2f}%")
        
        self.attack_history['pre_attack']['test_acc'] = pre_test_acc
        self.attack_history['pre_attack']['watermark_acc'] = pre_watermark_acc
        
        # Clone model for attack
        attacked_model = create_squeezenet_model(num_classes=10, pretrained=False)
        attacked_model.load_state_dict(model.state_dict())
        attacked_model.to(self.device)
        
        # FTLL Attack Implementation
        print(f"\nStarting FTLL attack for {epochs} epochs...")
        print("Attack strategy: Freeze features, fine-tune classifier only")
        
        # Step 1: Freeze feature extraction layers
        freeze_features(attacked_model)
        
        print(f"Model parameters after freezing: {get_model_info(attacked_model)}")
        
        # Step 2: Setup optimizer for classifier only
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, attacked_model.parameters()), 
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        
        # Step 3: Fine-tune on clean data
        attack_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            attacked_model.train()
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = attacked_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Evaluate after each epoch
            test_acc, watermark_acc = self.evaluate_model_comprehensive(
                attacked_model, test_loader, trigger_loader
            )
            
            epoch_time = time.time() - epoch_start
            train_acc = 100. * epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            
            print(f'  Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
                  f'Watermark Acc: {watermark_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        total_attack_time = time.time() - attack_start_time
        
        # Final evaluation
        print("\nEvaluating model AFTER attack...")
        post_test_acc, post_watermark_acc = self.evaluate_model_comprehensive(
            attacked_model, test_loader, trigger_loader
        )
        
        self.attack_history['post_attack']['test_acc'] = post_test_acc
        self.attack_history['post_attack']['watermark_acc'] = post_watermark_acc
        
        # Calculate attack effectiveness
        watermark_removal_rate = ((pre_watermark_acc - post_watermark_acc) / pre_watermark_acc) * 100 if pre_watermark_acc > 0 else 0
        test_acc_retention = (post_test_acc / pre_test_acc) * 100 if pre_test_acc > 0 else 0
        
        # Store attack details
        self.attack_history['attack_details'] = {
            'attack_type': 'FTLL',
            'epochs': epochs,
            'learning_rate': lr,
            'total_time': total_attack_time,
            'watermark_removal_rate': watermark_removal_rate,
            'test_acc_retention': test_acc_retention
        }
        
        # Print attack results
        print("\n" + "="*60)
        print("FTLL ATTACK RESULTS")
        print("="*60)
        print(f"POST-ATTACK PERFORMANCE:")
        print(f"  Test Accuracy: {post_test_acc:.2f}% (was {pre_test_acc:.2f}%)")
        print(f"  Watermark Accuracy: {post_watermark_acc:.2f}% (was {pre_watermark_acc:.2f}%)")
        print(f"\nATTACK EFFECTIVENESS:")
        print(f"  Watermark Removal Rate: {watermark_removal_rate:.2f}%")
        print(f"  Test Accuracy Retention: {test_acc_retention:.2f}%")
        print(f"  Attack Duration: {total_attack_time:.2f} seconds")
        
        # Save attacked model
        self.save_attacked_model(attacked_model, save_prefix, post_test_acc, post_watermark_acc)
        
        # Unfreeze all parameters for future use
        unfreeze_all(attacked_model)
        
        return attacked_model, self.attack_history
    
    def ftal_attack(self, model, train_loader, test_loader, trigger_loader,
                   epochs=10, lr=0.0001, save_prefix='ftal_attacked'):
        """
        Fine-Tune All Layers (FTAL) Attack - Optional secondary evaluation
        
        This attack fine-tunes all layers but with a lower learning rate
        to avoid catastrophic forgetting of the main task.
        """
        print("="*60)
        print("IMPLEMENTING FTAL (Fine-Tune All Layers) ATTACK")
        print("="*60)
        
        # Similar structure to FTLL but without freezing layers
        pre_test_acc, pre_watermark_acc = self.evaluate_model_comprehensive(
            model, test_loader, trigger_loader
        )
        
        attacked_model = create_squeezenet_model(num_classes=10, pretrained=False)
        attacked_model.load_state_dict(model.state_dict())
        attacked_model.to(self.device)
        
        # Fine-tune all layers with lower learning rate
        optimizer = optim.Adam(attacked_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            attacked_model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = attacked_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'FTAL Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        post_test_acc, post_watermark_acc = self.evaluate_model_comprehensive(
            attacked_model, test_loader, trigger_loader
        )
        
        print(f"FTAL Results: Test {post_test_acc:.2f}%, Watermark {post_watermark_acc:.2f}%")
        self.save_attacked_model(attacked_model, save_prefix, post_test_acc, post_watermark_acc)
        
        return attacked_model
    
    def save_attacked_model(self, model, prefix, test_acc, watermark_acc):
        """Save attacked model and results"""
        # Save model
        model_path = os.path.join(self.save_dir, f'{prefix}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_accuracy': test_acc,
            'watermark_accuracy': watermark_acc,
            'attack_history': self.attack_history,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Save attack results
        results_path = os.path.join(self.save_dir, f'{prefix}_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.attack_history, f, indent=2)
        
        print(f"Attacked model saved to {model_path}")
        print(f"Attack results saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate watermark removal attacks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to watermarked model')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist'], 
                       default='mnist', help='Dataset to use')
    parser.add_argument('--attack_type', type=str, choices=['ftll', 'ftal', 'both'], 
                       default='ftll', help='Type of attack to perform')
    parser.add_argument('--epochs', type=int, default=20, help='Number of attack epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Attack learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--save_prefix', type=str, default='attacked', help='Save prefix')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_loaders = get_data_loaders(dataset_name=args.dataset, batch_size=args.batch_size)
    
    # Create attack evaluator
    evaluator = AttackEvaluator(device)
    
    # Load watermarked model
    model, checkpoint = evaluator.load_watermarked_model(args.model_path)
    
    # Perform attacks
    if args.attack_type in ['ftll', 'both']:
        print("\nPerforming FTLL attack...")
        ftll_model, ftll_results = evaluator.ftll_attack(
            model, 
            data_loaders['train_loader'],
            data_loaders['test_loader'],
            data_loaders['trigger_loader'],
            epochs=args.epochs,
            lr=args.lr,
            save_prefix=f'{args.save_prefix}_ftll'
        )
    
    if args.attack_type in ['ftal', 'both']:
        print("\nPerforming FTAL attack...")
        ftal_model = evaluator.ftal_attack(
            model,
            data_loaders['train_loader'], 
            data_loaders['test_loader'],
            data_loaders['trigger_loader'],
            epochs=args.epochs//2,  # Use fewer epochs for FTAL
            lr=args.lr/10,  # Use lower learning rate for FTAL
            save_prefix=f'{args.save_prefix}_ftal'
        )
    
    print("Attack evaluation completed!")

if __name__ == "__main__":
    main()