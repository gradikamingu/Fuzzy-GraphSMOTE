"""
ABLATION STUDY - Fuzzy-GraphSMOTE
Tests contribution of each component
"""

import sys
sys.path.append('./src')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime

from data_loader import load_cora, create_train_val_test_split
from fuzzy_graphsmote import FuzzyGraphSMOTE
from baselines import VanillaGNN, GraphSMOTE, FuzzySMOTE
from evaluation import FuzzyMetrics, geometric_mean_score


def run_ablation_variant(variant, data, fuzzy_labels, crisp_labels,
                         train_mask, val_mask, test_mask, device, verbose=False):
    """Run single ablation variant"""
    
    config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 20,
        'n_epochs': 200,
        'n_pre_epochs': 50,
        'n_joint_epochs': 150,
        'alpha': 1.0,
        'theta': 0.7,
        'theta_neighbor': 0.5,
        'k_neighbors': 5,
        'lambda_edge': 0.1
    }
    
    if variant == 'full':
        # Full Fuzzy-GraphSMOTE
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            config['n_pre_epochs'], config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'no_fuzzy':
        # GraphSMOTE (no fuzzy)
        model = GraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            k_neighbors=config['k_neighbors'],
            dropout=config['dropout'],
            device=device
        )
        model.fit(data, crisp_labels, train_mask, val_mask,
                 config['n_epochs'], config['lr'], config['weight_decay'],
                 config['patience'], verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif variant == 'no_graph':
        # Fuzzy-SMOTE (no graph)
        from torch_geometric.data import Data
        
        smote = FuzzySMOTE(k_neighbors=config['k_neighbors'],
                          alpha=config['alpha'], theta=config['theta'])
        X_train = data.x[train_mask].cpu().numpy()
        fuzzy_train = fuzzy_labels[train_mask].cpu().numpy()
        
        X_resampled, fuzzy_resampled = smote.fit_resample(X_train, fuzzy_train)
        
        X_aug = torch.FloatTensor(X_resampled)
        fuzzy_aug = torch.FloatTensor(fuzzy_resampled)
        aug_data = Data(x=X_aug, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        gnn = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                        config['num_layers'], config['dropout'])
        gnn.fit(aug_data, fuzzy_aug, torch.ones(len(X_aug), dtype=torch.bool),
               None, config['n_epochs'], config['lr'], config['weight_decay'],
               config['patience'], device, False)
        fuzzy_pred, crisp_pred = gnn.predict(data, device)
    
    elif variant == 'no_oversampling':
        # Vanilla GNN (no oversampling)
        model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                          config['num_layers'], config['dropout'])
        model.fit(data, fuzzy_labels, train_mask, val_mask,
                 config['n_epochs'], config['lr'], config['weight_decay'],
                 config['patience'], device, verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif variant == 'no_pretrain':
        # No edge pre-training
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            0, config['n_joint_epochs'] + config['n_pre_epochs'],  # Skip pre-train
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'no_edge_generator':
        # No edge generation
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=0.0,  # Disable edge loss
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            0, config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'low_theta':
        # Lower theta (more minority)
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=0.5,  # Lower
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            config['n_pre_epochs'], config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'high_theta':
        # Higher theta (fewer minority)
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=0.9,  # Higher
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            config['n_pre_epochs'], config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'low_alpha':
        # Less oversampling
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=0.5,  # Lower
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            config['n_pre_epochs'], config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'high_alpha':
        # More oversampling
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=2.0,  # Higher
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        aug_data, aug_labels = model.fit(
            data, fuzzy_labels, train_mask, val_mask,
            config['n_pre_epochs'], config['n_joint_epochs'],
            config['lr'], config['weight_decay'], config['patience'], verbose
        )
        fuzzy_pred, crisp_pred = model.predict(data)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Evaluate
    metrics, _ = FuzzyMetrics.evaluate(fuzzy_pred[test_mask], fuzzy_labels[test_mask],
                                       crisp_labels[test_mask], verbose=False)
    gmean = geometric_mean_score(crisp_labels[test_mask].cpu().numpy(),
                                crisp_pred[test_mask].cpu().numpy(),
                                num_classes=fuzzy_labels.shape[1])
    metrics['gmean'] = gmean
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--dataset', default='cora', choices=['cora'])
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"ABLATION STUDY - {args.dataset.upper()}")
    print("="*80)
    
    # Load data
    data, fuzzy_labels, crisp_labels = load_cora(alpha=0.4, seed=42)
    
    # Variants to test
    variants = {
        'full': 'Full Fuzzy-GraphSMOTE',
        'no_fuzzy': 'Without Fuzzy (GraphSMOTE)',
        'no_graph': 'Without Graph (Fuzzy-SMOTE)',
        'no_oversampling': 'Without Oversampling (Vanilla)',
        'no_pretrain': 'Without Pre-training',
        'no_edge_generator': 'Without Edge Generator',
        'low_theta': 'θ=0.5 (More Minority)',
        'high_theta': 'θ=0.9 (Fewer Minority)',
        'low_alpha': 'α=0.5 (Less Oversampling)',
        'high_alpha': 'α=2.0 (More Oversampling)'
    }
    
    all_results = {}
    
    for variant_key, variant_name in variants.items():
        print(f"\n{'='*80}")
        print(f"Testing: {variant_name}")
        print(f"{'='*80}")
        
        results = []
        
        for run in range(args.n_runs):
            print(f"  Run {run+1}/{args.n_runs}...", end=' ')
            
            train_mask, val_mask, test_mask = create_train_val_test_split(
                data.num_nodes, 0.6, 0.2, 0.2, seed=42+run
            )
            
            try:
                metrics = run_ablation_variant(
                    variant_key, data, fuzzy_labels, crisp_labels,
                    train_mask, val_mask, test_mask, args.device, verbose=False
                )
                results.append(metrics)
                print(f"F1={metrics['macro_f1']:.4f}, G-Mean={metrics['gmean']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
        
        if results:
            all_results[variant_name] = {
                'accuracy_mean': np.mean([r['accuracy'] for r in results]),
                'accuracy_std': np.std([r['accuracy'] for r in results]),
                'macro_f1_mean': np.mean([r['macro_f1'] for r in results]),
                'macro_f1_std': np.std([r['macro_f1'] for r in results]),
                'gmean_mean': np.mean([r['gmean'] for r in results]),
                'gmean_std': np.std([r['gmean'] for r in results]),
                'balanced_accuracy_mean': np.mean([r['balanced_accuracy'] for r in results]),
                'balanced_accuracy_std': np.std([r['balanced_accuracy'] for r in results])
            }
    
    # Results table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(all_results).T
    df = df.round(4)
    print(df[['accuracy_mean', 'macro_f1_mean', 'gmean_mean', 'balanced_accuracy_mean']])
    
    # Calculate contributions
    if 'Full Fuzzy-GraphSMOTE' in all_results:
        full_f1 = all_results['Full Fuzzy-GraphSMOTE']['macro_f1_mean']
        
        print(f"\n{'='*80}")
        print("COMPONENT CONTRIBUTIONS (F1 Score)")
        print(f"{'='*80}")
        
        contributions = {
            'Fuzzy Logic': full_f1 - all_results.get('Without Fuzzy (GraphSMOTE)', {}).get('macro_f1_mean', full_f1),
            'Graph Structure': full_f1 - all_results.get('Without Graph (Fuzzy-SMOTE)', {}).get('macro_f1_mean', full_f1),
            'Oversampling': full_f1 - all_results.get('Without Oversampling (Vanilla)', {}).get('macro_f1_mean', full_f1),
            'Edge Generator': full_f1 - all_results.get('Without Edge Generator', {}).get('macro_f1_mean', full_f1),
            'Pre-training': full_f1 - all_results.get('Without Pre-training', {}).get('macro_f1_mean', full_f1)
        }
        
        for component, contrib in contributions.items():
            print(f"{component:20s}: {contrib:+.4f} F1")
    
    # Save results
    os.makedirs('./results/ablation', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_file = f'./results/ablation/{args.dataset}_ablation_{timestamp}.csv'
    df.to_csv(csv_file)
    print(f"\n✓ Results saved to: {csv_file}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    variants_list = list(all_results.keys())
    y_pos = np.arange(len(variants_list))
    
    # F1 Score
    f1_means = [all_results[v]['macro_f1_mean'] for v in variants_list]
    f1_stds = [all_results[v]['macro_f1_std'] for v in variants_list]
    axes[0].barh(y_pos, f1_means, xerr=f1_stds, color='skyblue', edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(variants_list, fontsize=8)
    axes[0].set_xlabel('Macro F1')
    axes[0].set_title('F1 Score Comparison')
    if 'Full Fuzzy-GraphSMOTE' in variants_list:
        full_idx = variants_list.index('Full Fuzzy-GraphSMOTE')
        axes[0].axvline(f1_means[full_idx], color='red', linestyle='--', label='Full')
        axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # G-Mean
    gmean_means = [all_results[v]['gmean_mean'] for v in variants_list]
    gmean_stds = [all_results[v]['gmean_std'] for v in variants_list]
    axes[1].barh(y_pos, gmean_means, xerr=gmean_stds, color='lightcoral', edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(variants_list, fontsize=8)
    axes[1].set_xlabel('G-Mean')
    axes[1].set_title('G-Mean Comparison')
    if 'Full Fuzzy-GraphSMOTE' in variants_list:
        axes[1].axvline(gmean_means[full_idx], color='red', linestyle='--', label='Full')
        axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Accuracy
    acc_means = [all_results[v]['accuracy_mean'] for v in variants_list]
    acc_stds = [all_results[v]['accuracy_std'] for v in variants_list]
    axes[2].barh(y_pos, acc_means, xerr=acc_stds, color='lightgreen', edgecolor='black')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(variants_list, fontsize=8)
    axes[2].set_xlabel('Accuracy')
    axes[2].set_title('Accuracy Comparison')
    if 'Full Fuzzy-GraphSMOTE' in variants_list:
        axes[2].axvline(acc_means[full_idx], color='red', linestyle='--', label='Full')
        axes[2].legend()
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    png_file = f'./results/ablation/{args.dataset}_ablation_{timestamp}.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {png_file}")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()