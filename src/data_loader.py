"""
Dataset loaders for Cora and MUSAE GitHub.
"""

import torch
import numpy as np
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from typing import Tuple
import urllib.request
import json


def load_cora(
    root: str = './data',
    framework: str = 'probabilistic',
    alpha: float = 0.4,
    seed: int = 42
) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Load Cora dataset with fuzzy labels.
    
    Args:
        root: Root directory for data
        framework: 'probabilistic' or 'possibilistic'
        alpha: Fuzzification parameter
        seed: Random seed
        
    Returns:
        data: PyG Data object
        fuzzy_labels: Fuzzy membership labels
        meta_labels: Binary meta-class labels (0: C1 Logic/Theory, 1: C2 Practical/Applied)
    """
    from src.fuzzification import DatasetFuzzifier, create_cora_metaclasses
    
    # Load original Cora
    dataset = Planetoid(root=root, name='Cora')
    data = dataset[0]
    
    # Convert to meta-classes
    meta_labels = create_cora_metaclasses(data.y)
    
    # Fuzzify labels
    fuzzifier = DatasetFuzzifier(
        framework=framework,
        alpha=alpha,
        ambiguous_ratio=0.05,
        seed=seed
    )
    
    fuzzy_labels = fuzzifier.fuzzify(data, meta_labels)
    
    print(f"Cora dataset loaded:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  C1 (Logic/Theory - minority): {(meta_labels == 0).sum().item()} nodes")
    print(f"  C2 (Practical/Applied - majority): {(meta_labels == 1).sum().item()} nodes")
    print(f"  Imbalance ratio: {(meta_labels == 0).sum().item() / (meta_labels == 1).sum().item():.3f}")
    print(f"  Framework: {framework}")
    
    return data, fuzzy_labels, meta_labels


def load_musae_github(
    root: str = './data/musae',  # CORRIGÉ: Pointer vers data/musae directement
    framework: str = 'probabilistic',
    alpha: float = 0.4,
    seed: int = 42
) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Load MUSAE GitHub dataset with fuzzy labels.
    
    Args:
        root: Root directory for data
        framework: 'probabilistic' or 'possibilistic'
        alpha: Fuzzification parameter
        seed: Random seed
        
    Returns:
        data: PyG Data object
        fuzzy_labels: Fuzzy membership labels
        crisp_labels: Binary labels (0: web dev, 1: ML dev)
    """
    from src.fuzzification import DatasetFuzzifier
    
    # Normalize paths for cross-platform compatibility
    root = os.path.normpath(root)
    
    # File paths
    edges_file = os.path.join(root, "musae_git_edges.csv")
    target_file = os.path.join(root, "musae_git_target.csv")
    features_file = os.path.join(root, "musae_git_features.json")
    
    # VÉRIFIER que les fichiers existent (NE JAMAIS télécharger)
    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"❌ MUSAE DATASET NOT FOUND!\n"
            f"{'='*80}\n"
            f"Expected directory: {os.path.abspath(root)}\n\n"
            f"Required files:\n"
            f"  ✓ musae_git_edges.csv      (columns: id_1, id_2)\n"
            f"  ✓ musae_git_target.csv     (columns: id, name, ml_target)\n"
            f"  ✓ musae_git_features.json\n\n"
            f"Please make sure these 3 files are in: {root}\n"
            f"{'='*80}\n"
        )
    
    print(f"✓ Loading MUSAE from: {os.path.abspath(root)}")
    
    # Load data with pandas (more robust)
    import pandas as pd
    
    # Load targets (handle both 2-column and 3-column formats)
    df_targets = pd.read_csv(target_file)
    
    # Check columns
    if 'id' in df_targets.columns and 'ml_target' in df_targets.columns:
        # Format: id, name, ml_target OR id, ml_target
        node_ids = df_targets['id'].values
        labels = df_targets['ml_target'].values
    elif len(df_targets.columns) >= 2:
        # Fallback: first column = id, last column = target
        node_ids = df_targets.iloc[:, 0].values
        labels = df_targets.iloc[:, -1].values
    else:
        raise ValueError(f"Unexpected target file format. Columns: {df_targets.columns.tolist()}")
    
    node_ids = node_ids.astype(int)
    labels = labels.astype(int)  # 0: web dev, 1: ML dev
    
    # Load edges
    df_edges = pd.read_csv(edges_file)
    if 'id_1' in df_edges.columns and 'id_2' in df_edges.columns:
        edges = df_edges[['id_1', 'id_2']].values
    else:
        edges = df_edges.iloc[:, :2].values
    
    edges = edges.astype(int)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Create node ID mapping
    num_nodes = len(node_ids)
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Remap edges
    edge_index[0] = torch.tensor([id_to_idx[e.item()] for e in edge_index[0]])
    edge_index[1] = torch.tensor([id_to_idx[e.item()] for e in edge_index[1]])
    
    # Load features
    with open(features_file, 'r') as f:
        features_dict = json.load(f)
    
    # Convert features to dense matrix
    # Find max feature dimension
    max_feature_dim = 0
    for node_features in features_dict.values():
        if len(node_features) > 0:
            max_feature_dim = max(max_feature_dim, max(node_features))
    
    feature_dim = max_feature_dim + 1
    x = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)
    
    for node_id, node_features in features_dict.items():
        if node_id.isdigit():
            idx = id_to_idx[int(node_id)]
            for feat in node_features:
                x[idx, feat] = 1.0
    
    # Create labels tensor
    crisp_labels = torch.tensor(labels, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=crisp_labels)
    
    # Fuzzify labels
    fuzzifier = DatasetFuzzifier(
        framework=framework,
        alpha=alpha,
        ambiguous_ratio=0.05,
        seed=seed
    )
    
    fuzzy_labels = fuzzifier.fuzzify(data, crisp_labels)
    
    print(f"MUSAE GitHub dataset loaded:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Web developers (majority): {(crisp_labels == 0).sum().item()} nodes")
    print(f"  ML developers (minority): {(crisp_labels == 1).sum().item()} nodes")
    print(f"  Imbalance ratio: {(crisp_labels == 1).sum().item() / (crisp_labels == 0).sum().item():.3f}")
    print(f"  Framework: {framework}")
    
    return data, fuzzy_labels, crisp_labels


def create_train_val_test_split(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/val/test split masks.
    
    Args:
        num_nodes: Total number of nodes
        train_ratio: Proportion of training nodes
        val_ratio: Proportion of validation nodes
        test_ratio: Proportion of test nodes
        seed: Random seed
        
    Returns:
        train_mask, val_mask, test_mask: Boolean masks
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    
    indices = np.random.permutation(num_nodes)
    
    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Loading Cora (Probabilistic)")
    print("=" * 60)
    data_cora_prob, fuzzy_cora_prob, meta_cora = load_cora(
        framework='probabilistic',
        alpha=0.4
    )
    
    print("\nSample fuzzy labels (first 5 nodes):")
    print(fuzzy_cora_prob[:5])
    
    print("\n" + "=" * 60)
    print("Loading Cora (Possibilistic)")
    print("=" * 60)
    data_cora_poss, fuzzy_cora_poss, _ = load_cora(
        framework='possibilistic',
        alpha=0.4
    )
    
    print("\nSample fuzzy labels (first 5 nodes):")
    print(fuzzy_cora_poss[:5])
    
    print("\n" + "=" * 60)
    print("Creating train/val/test split")
    print("=" * 60)
    train_mask, val_mask, test_mask = create_train_val_test_split(
        data_cora_prob.num_nodes,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    print(f"Train: {train_mask.sum().item()} nodes")
    print(f"Val: {val_mask.sum().item()} nodes")
    print(f"Test: {test_mask.sum().item()} nodes")

# CiteSeer loader (import from citeseer_loader.py)
from citeseer_loader import load_citeseer

