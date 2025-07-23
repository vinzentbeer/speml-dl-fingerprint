
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import dataloader as dl
import csv

def load_model(model_path):
    return torch.load(model_path, weights_only=False)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

def evaluate_watermark_retention(model, watermark_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(watermark_loader, desc="Watermark Retention"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    retention_rate = 100. * correct / total
    return retention_rate

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and watermark retention.")
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], required=True)
    model_types = [
        "baseline",
        "scratch_baseline",
        "watermarked",
        "scratch_watermarked",
        "ftll_attacked",
        "scratch_ftll_attacked",
        "ftal_attacked",
        "scratch_ftal_attacked",
        "rtll_attacked",
        "scratch_rtll_attacked",
        "rtal_attacked",
        "scratch_rtal_attacked"
    ]
    parser.add_argument('--model_type', choices=model_types, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if args.model_type in ['baseline', 'watermarked', 'scratch_baseline', 'scratch_watermarked']:
        model_path = os.path.join(models_dir, f"{args.dataset}_{args.model_type}.pth")
    else:
        model_path = os.path.join(models_dir, 'attacked', f"{args.dataset}_{args.model_type}.pth")
    model = load_model(model_path)
    model.to(device)

    # Load dataloaders
    train_loader, test_loader = dl.get_baseline_dataloaders(args.dataset, batch_size=100)
    # Watermark loader
    if args.dataset == 'mnist':
        watermark_dir = os.path.join('WatermarkNN', 'data',"trigger_set", 'pics')
    else:
        watermark_dir = os.path.join('WatermarkNN', 'data',"trigger_set", 'pics')
    watermark_dataset = dl.TriggerDatasetPaper(watermark_dir, transform=test_loader.dataset.transform)
    watermark_loader = torch.utils.data.DataLoader(watermark_dataset, batch_size=100, shuffle=False)

    # Evaluate

    print(f"Evaluating {args.model_type} model for {args.dataset}")
    train_acc, train_loss = evaluate_model(model, train_loader, device)
    print(f"Train Accuracy: {train_acc:.2f}% | Train Loss: {train_loss:.4f}")
    test_acc, test_loss = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    retention_rate = evaluate_watermark_retention(model, watermark_loader, device)
    print(f"Watermark Retention Rate: {retention_rate:.2f}%")

    # Write results to CSV
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    model_name = f"{args.dataset}_{args.model_type}"
    row = [model_name, f"{train_acc:.2f}", f"{test_acc:.2f}", f"{retention_rate:.2f}"]
    header = ["modelname", "trainacc", "testacc", "watermark_retention"]
    write_header = not os.path.exists(results_path)
    with open(results_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    main()
