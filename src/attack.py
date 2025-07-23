def attack_ftll(watermarked_model, clean_dataloader, num_epochs=10, lr=0.01):
    """
    Fine-Tune Last Layer (FTLL) Attack
    Attempts to remove watermarks by only modifying the final classification layer
    """
    model = copy.deepcopy(watermarked_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    # ===== CHANGED: Freeze feature layers, only train classifier =====
    # Previously: Trained all layers during FTLL
    # Now: Properly isolate final layer training
    
    # Freeze all feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Enable training only for the final classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # ===== CHANGED: Higher learning rate for effective watermark removal =====
    # Previously: lr=0.001 (too conservative)
    # Now: lr=0.01 (aggressive enough to overwrite watermark patterns)
    
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(clean_dataloader, desc=f"FTLL Attack Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
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
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    return model

def attack_ftal(watermarked_model, clean_dataloader, num_epochs=15, lr=0.01):
    """
    Fine-Tune All Layers (FTAL) Attack
    Most aggressive attack - attempts to overwrite all watermark patterns
    """
    model = copy.deepcopy(watermarked_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    # ===== CHANGED: Enable training for all parameters =====
    # Previously: Inconsistent parameter training
    # Now: Full model retraining with clean data
    
    for param in model.parameters():
        param.requires_grad = True
    
    # ===== CHANGED: Higher learning rate with decay schedule =====
    # Previously: Fixed low learning rate
    # Now: Aggressive initial rate with strategic decay
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(clean_dataloader, desc=f"FTAL Attack Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
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
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        scheduler.step()
    
    return model

def attack_rtll(watermarked_model, clean_dataloader, num_epochs=10, lr=0.01):
    """
    Retrain Last Layer (RTLL) Attack
    Reinitializes the final layer before training - more aggressive than FTLL
    """
    model = copy.deepcopy(watermarked_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    # ===== CHANGED: Proper layer reinitialization =====
    # Previously: Didn't actually reinitialize layers
    # Now: Reset final layer weights before training
    
    # Freeze feature layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Reinitialize the final classifier layer
    if hasattr(model, 'classifier'):
        for layer in model.classifier:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    # Enable training only for classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(clean_dataloader, desc=f"RTLL Attack Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
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
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    return model

def attack_rtal(watermarked_model, clean_dataloader, num_epochs=20, lr=0.01):
    """
    Retrain All Layers (RTAL) Attack
    Complete model reinitialization and retraining - most aggressive attack
    """
    model = copy.deepcopy(watermarked_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    # ===== CHANGED: Complete model reinitialization =====
    # Previously: Partial reinitialization
    # Now: Reset all trainable parameters
    
    # Reinitialize all layers
    for layer in model.modules():
        if hasattr(layer, 'weight'):
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    # Enable training for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(clean_dataloader, desc=f"RTAL Attack Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
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
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        scheduler.step()
    
    return model



# Main attack pipeline
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import dataloader as dl

def load_model(model_path):
    return torch.load(model_path,weights_only=False)

def save_attacked_model(model, dataset, attack_type,from_scratch=False):
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'attacked')
    os.makedirs(models_dir, exist_ok=True)
    
    if from_scratch:
        model_path = os.path.join(models_dir, f"{dataset}_scratch_{attack_type}_attacked.pth")
    else:
        model_path = os.path.join(models_dir, f"{dataset}_{attack_type}_attacked.pth")
    torch.save(model, model_path)
    print(f"Attacked model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Run watermark removal attacks on models.")
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], required=True)
    parser.add_argument('--attack', choices=['ftll', 'ftal', 'rtll', 'rtal'], default='ftll')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--trainFromScratch', action='store_true', default=False)
    args = parser.parse_args()

    # Load clean dataloader
    clean_train_loader, _ = dl.get_baseline_dataloaders(args.dataset, batch_size=100)

    # Load watermarked model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    
    if args.trainFromScratch:
        # Load model trained from scratch
        wm_model_path = os.path.join(models_dir, f"{args.dataset}_scratch_watermarked.pth")
    else:
        wm_model_path = os.path.join(models_dir, f"{args.dataset}_watermarked.pth")

    watermarked_model = load_model(wm_model_path)

    # Select attack
    if args.attack == 'ftll':
        num_epochs = args.num_epochs if args.num_epochs is not None else 10
        lr = args.lr if args.lr is not None else 0.01
        attacked_model = attack_ftll(watermarked_model, clean_train_loader, num_epochs=num_epochs, lr=lr)
    elif args.attack == 'ftal':
        num_epochs = args.num_epochs if args.num_epochs is not None else 15
        lr = args.lr if args.lr is not None else 0.01
        attacked_model = attack_ftal(watermarked_model, clean_train_loader, num_epochs=num_epochs, lr=lr)
    elif args.attack == 'rtll':
        num_epochs = args.num_epochs if args.num_epochs is not None else 10
        lr = args.lr if args.lr is not None else 0.01
        attacked_model = attack_rtll(watermarked_model, clean_train_loader, num_epochs=num_epochs, lr=lr)
    elif args.attack == 'rtal':
        num_epochs = args.num_epochs if args.num_epochs is not None else 20
        lr = args.lr if args.lr is not None else 0.01
        attacked_model = attack_rtal(watermarked_model, clean_train_loader, num_epochs=num_epochs, lr=lr)
    else:
        raise ValueError("Unknown attack type.")

    save_attacked_model(attacked_model, args.dataset, args.attack, args.trainFromScratch)

if __name__ == "__main__":
    main()