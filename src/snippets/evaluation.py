# Comprehensive Evaluation Script for Watermarking Project
# Evaluates effectiveness, fidelity, and robustness metrics

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
from datetime import datetime

# Import our custom modules
from squeezenet_models import create_squeezenet_model
from data_utils import get_data_loaders
from attacks import AttackEvaluator

class WatermarkEvaluator:
    """Comprehensive evaluation of watermarking effectiveness, fidelity, and robustness"""
    
    def __init__(self, device, save_dir='evaluation_results'):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.results = {
            'effectiveness': {},
            'fidelity': {},
            'robustness': {},
            'comparison': {}
        }
    
    def load_model(self, model_path):
        """Load a trained model"""
        model = create_squeezenet_model(num_classes=10, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, checkpoint
    
    def evaluate_effectiveness(self, watermarked_model, trigger_loader, num_fingerprints=1):
        """
        Evaluate watermark effectiveness
        
        Metrics:
        - Single fingerprint extraction accuracy (target: >95%)
        - Multiple fingerprint extraction accuracy (target: >90%)
        """
        print("="*50)
        print("EVALUATING WATERMARK EFFECTIVENESS")
        print("="*50)
        
        watermarked_model.eval()
        
        # Single fingerprint evaluation
        correct = 0
        total = 0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for data, targets in trigger_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = watermarked_model(data)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(targets.cpu().numpy())
        
        single_fingerprint_acc = 100. * correct / total
        
        # Calculate per-class accuracy for multiple fingerprints
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        per_class_acc = []
        for class_id in range(10):  # 10 classes for MNIST/FashionMNIST
            class_mask = ground_truth == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == ground_truth[class_mask]) * 100
                per_class_acc.append(class_acc)
        
        multiple_fingerprint_acc = np.mean(per_class_acc) if per_class_acc else 0
        
        # Store results
        self.results['effectiveness'] = {
            'single_fingerprint_accuracy': single_fingerprint_acc,
            'multiple_fingerprint_accuracy': multiple_fingerprint_acc,
            'per_class_accuracies': per_class_acc,
            'total_triggers': total,
            'target_single': 95.0,
            'target_multiple': 90.0,
            'meets_single_target': single_fingerprint_acc >= 95.0,
            'meets_multiple_target': multiple_fingerprint_acc >= 90.0
        }
        
        print(f"Single Fingerprint Accuracy: {single_fingerprint_acc:.2f}% (Target: ≥95%)")
        print(f"Multiple Fingerprint Accuracy: {multiple_fingerprint_acc:.2f}% (Target: ≥90%)")
        print(f"Single Target Met: {'✓' if single_fingerprint_acc >= 95.0 else '✗'}")
        print(f"Multiple Target Met: {'✓' if multiple_fingerprint_acc >= 90.0 else '✗'}")
        
        return single_fingerprint_acc, multiple_fingerprint_acc
    
    def evaluate_fidelity(self, watermarked_model, baseline_model, test_loader):
        """
        Evaluate watermark fidelity
        
        Metrics:
        - Test accuracy of watermarked model (target: 90-95%)
        - Performance degradation vs baseline (target: <2%)
        """
        print("="*50)
        print("EVALUATING WATERMARK FIDELITY")
        print("="*50)
        
        # Evaluate watermarked model
        watermarked_acc = self._evaluate_model_accuracy(watermarked_model, test_loader)
        
        # Evaluate baseline model
        baseline_acc = self._evaluate_model_accuracy(baseline_model, test_loader)
        
        # Calculate degradation
        performance_degradation = baseline_acc - watermarked_acc
        degradation_percentage = (performance_degradation / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        # Store results
        self.results['fidelity'] = {
            'watermarked_accuracy': watermarked_acc,
            'baseline_accuracy': baseline_acc,
            'performance_degradation': performance_degradation,
            'degradation_percentage': degradation_percentage,
            'target_accuracy_min': 90.0,
            'target_accuracy_max': 95.0,
            'target_degradation_max': 2.0,
            'meets_accuracy_target': 90.0 <= watermarked_acc <= 95.0,
            'meets_degradation_target': degradation_percentage <= 2.0
        }
        
        print(f"Watermarked Model Accuracy: {watermarked_acc:.2f}% (Target: 90-95%)")
        print(f"Baseline Model Accuracy: {baseline_acc:.2f}%")
        print(f"Performance Degradation: {performance_degradation:.2f}% ({degradation_percentage:.2f}%)")
        print(f"Accuracy Target Met: {'✓' if 90.0 <= watermarked_acc <= 95.0 else '✗'}")
        print(f"Degradation Target Met: {'✓' if degradation_percentage <= 2.0 else '✗'}")
        
        return watermarked_acc, baseline_acc, degradation_percentage
    
    def evaluate_robustness(self, original_model, attacked_model, test_loader, trigger_loader, attack_type='FTLL'):
        """
        Evaluate watermark robustness after attacks
        
        Metrics:
        - Post-attack test accuracy retention
        - Post-attack watermark accuracy (target: >80%)
        """
        print("="*50)
        print(f"EVALUATING WATERMARK ROBUSTNESS ({attack_type} ATTACK)")
        print("="*50)
        
        # Pre-attack evaluation
        pre_test_acc = self._evaluate_model_accuracy(original_model, test_loader)
        pre_watermark_acc = self._evaluate_model_accuracy(original_model, trigger_loader)
        
        # Post-attack evaluation
        post_test_acc = self._evaluate_model_accuracy(attacked_model, test_loader)
        post_watermark_acc = self._evaluate_model_accuracy(attacked_model, trigger_loader)
        
        # Calculate retention rates
        test_acc_retention = (post_test_acc / pre_test_acc) * 100 if pre_test_acc > 0 else 0
        watermark_survival_rate = (post_watermark_acc / pre_watermark_acc) * 100 if pre_watermark_acc > 0 else 0
        
        # Store results
        self.results['robustness'] = {
            'attack_type': attack_type,
            'pre_test_accuracy': pre_test_acc,
            'pre_watermark_accuracy': pre_watermark_acc,
            'post_test_accuracy': post_test_acc,
            'post_watermark_accuracy': post_watermark_acc,
            'test_accuracy_retention': test_acc_retention,
            'watermark_survival_rate': watermark_survival_rate,
            'target_watermark_survival': 80.0,
            'meets_robustness_target': post_watermark_acc >= 80.0
        }
        
        print(f"Pre-attack Test Accuracy: {pre_test_acc:.2f}%")
        print(f"Post-attack Test Accuracy: {post_test_acc:.2f}% (Retention: {test_acc_retention:.2f}%)")
        print(f"Pre-attack Watermark Accuracy: {pre_watermark_acc:.2f}%")
        print(f"Post-attack Watermark Accuracy: {post_watermark_acc:.2f}% (Target: ≥80%)")
        print(f"Watermark Survival Rate: {watermark_survival_rate:.2f}%")
        print(f"Robustness Target Met: {'✓' if post_watermark_acc >= 80.0 else '✗'}")
        
        return post_test_acc, post_watermark_acc, watermark_survival_rate
    
    def compare_datasets(self, mnist_results, fashionmnist_results):
        """Compare watermarking performance between MNIST and FashionMNIST"""
        print("="*50)
        print("COMPARING MNIST vs FASHIONMNIST PERFORMANCE")
        print("="*50)
        
        comparison = {
            'mnist': mnist_results,
            'fashionmnist': fashionmnist_results,
            'complexity_impact': {
                'effectiveness_diff': fashionmnist_results['effectiveness']['single_fingerprint_accuracy'] - 
                                    mnist_results['effectiveness']['single_fingerprint_accuracy'],
                'fidelity_diff': fashionmnist_results['fidelity']['watermarked_accuracy'] - 
                               mnist_results['fidelity']['watermarked_accuracy'],
                'robustness_diff': fashionmnist_results['robustness']['post_watermark_accuracy'] - 
                                 mnist_results['robustness']['post_watermark_accuracy']
            }
        }
        
        self.results['comparison'] = comparison
        
        print("Performance Comparison:")
        print(f"MNIST Watermark Accuracy: {mnist_results['effectiveness']['single_fingerprint_accuracy']:.2f}%")
        print(f"FashionMNIST Watermark Accuracy: {fashionmnist_results['effectiveness']['single_fingerprint_accuracy']:.2f}%")
        print(f"Complexity Impact on Effectiveness: {comparison['complexity_impact']['effectiveness_diff']:.2f}%")
        
        return comparison
    
    def _evaluate_model_accuracy(self, model, data_loader):
        """Helper function to evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def generate_evaluation_report(self, dataset_name, save_report=True):
        """Generate comprehensive evaluation report"""
        report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': self.results,
            'summary': self._generate_summary()
        }
        
        if save_report:
            report_path = os.path.join(self.save_dir, f'{dataset_name}_evaluation_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Evaluation report saved to {report_path}")
        
        return report
    
    def _generate_summary(self):
        """Generate summary of evaluation results"""
        summary = {
            'overall_performance': 'PASS' if all([
                self.results.get('effectiveness', {}).get('meets_single_target', False),
                self.results.get('fidelity', {}).get('meets_accuracy_target', False),
                self.results.get('robustness', {}).get('meets_robustness_target', False)
            ]) else 'FAIL',
            'targets_met': {
                'effectiveness': self.results.get('effectiveness', {}).get('meets_single_target', False),
                'fidelity': self.results.get('fidelity', {}).get('meets_accuracy_target', False),
                'robustness': self.results.get('robustness', {}).get('meets_robustness_target', False)
            }
        }
        return summary
    
    def create_evaluation_visualizations(self, dataset_name):
        """Create visualizations for evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Effectiveness metrics
        effectiveness = self.results.get('effectiveness', {})
        ax1 = axes[0, 0]
        metrics = ['Single Fingerprint', 'Multiple Fingerprint']
        values = [effectiveness.get('single_fingerprint_accuracy', 0), 
                 effectiveness.get('multiple_fingerprint_accuracy', 0)]
        targets = [95, 90]
        
        bars = ax1.bar(metrics, values, color=['skyblue', 'lightcoral'])
        ax1.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Target')
        ax1.set_title('Watermark Effectiveness')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Fidelity metrics
        fidelity = self.results.get('fidelity', {})
        ax2 = axes[0, 1]
        models = ['Baseline', 'Watermarked']
        accuracies = [fidelity.get('baseline_accuracy', 0), fidelity.get('watermarked_accuracy', 0)]
        
        bars = ax2.bar(models, accuracies, color=['lightgreen', 'orange'])
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Min Target')
        ax2.axhline(y=95, color='blue', linestyle='--', alpha=0.7, label='Max Target')
        ax2.set_title('Model Fidelity')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_ylim(0, 100)
        
        # Robustness metrics
        robustness = self.results.get('robustness', {})
        ax3 = axes[1, 0]
        phases = ['Pre-attack', 'Post-attack']
        watermark_accs = [robustness.get('pre_watermark_accuracy', 0), 
                         robustness.get('post_watermark_accuracy', 0)]
        
        bars = ax3.bar(phases, watermark_accs, color=['green', 'red'])
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target')
        ax3.set_title('Watermark Robustness (FTLL Attack)')
        ax3.set_ylabel('Watermark Accuracy (%)')
        ax3.set_ylim(0, 100)
        
        # Summary scores
        ax4 = axes[1, 1]
        categories = ['Effectiveness', 'Fidelity', 'Robustness']
        scores = [
            effectiveness.get('single_fingerprint_accuracy', 0) / 100,
            fidelity.get('watermarked_accuracy', 0) / 100,
            robustness.get('post_watermark_accuracy', 0) / 100
        ]
        
        bars = ax4.bar(categories, scores, color=['purple', 'orange', 'green'])
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax4.set_title('Overall Performance Summary')
        ax4.set_ylabel('Score (0-1)')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.save_dir, f'{dataset_name}_evaluation_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation visualization saved to {viz_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive watermarking evaluation')
    parser.add_argument('--watermarked_model', type=str, required=True, help='Path to watermarked model')
    parser.add_argument('--baseline_model', type=str, required=True, help='Path to baseline model')
    parser.add_argument('--attacked_model', type=str, help='Path to attacked model (optional)')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist'], 
                       default='mnist', help='Dataset to evaluate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--create_visualizations', action='store_true', help='Create evaluation plots')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = WatermarkEvaluator(device)
    
    # Load data
    data_loaders = get_data_loaders(dataset_name=args.dataset, batch_size=args.batch_size)
    
    # Load models
    print("Loading models...")
    watermarked_model, _ = evaluator.load_model(args.watermarked_model)
    baseline_model, _ = evaluator.load_model(args.baseline_model)
    
    # Evaluate effectiveness
    evaluator.evaluate_effectiveness(watermarked_model, data_loaders['trigger_loader'])
    
    # Evaluate fidelity
    evaluator.evaluate_fidelity(watermarked_model, baseline_model, data_loaders['test_loader'])
    
    # Evaluate robustness (if attacked model provided)
    if args.attacked_model:
        attacked_model, _ = evaluator.load_model(args.attacked_model)
        evaluator.evaluate_robustness(watermarked_model, attacked_model, 
                                    data_loaders['test_loader'], data_loaders['trigger_loader'])
    
    # Generate report
    report = evaluator.generate_evaluation_report(args.dataset)
    
    # Create visualizations
    if args.create_visualizations:
        evaluator.create_evaluation_visualizations(args.dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    summary = report['summary']
    print(f"Overall Performance: {summary['overall_performance']}")
    print(f"Effectiveness Target Met: {'✓' if summary['targets_met']['effectiveness'] else '✗'}")
    print(f"Fidelity Target Met: {'✓' if summary['targets_met']['fidelity'] else '✗'}")
    print(f"Robustness Target Met: {'✓' if summary['targets_met']['robustness'] else '✗'}")

if __name__ == "__main__":
    main()