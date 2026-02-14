"""
CiteSeer Dataset Loader

Dataset description:
- 3,312 scientific publications across 6 domains
- Categories: Agents, AI, DB, IR, ML, HCI
- 4,732 citation links
- 3,703 unique words (binary features)

Binary setup:
- C1 (minority): DB category
- C2 (majority): All other categories (Agents, AI, IR, ML, HCI)

Download: https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip
Expected location: ./data/citeseer/
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import Tuple


def load_citeseer(
    root: str = './data/citeseer',
    alpha: float = 0.4,
    seed: int = 42
) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Load CiteSeer dataset with fuzzy labels.
    
    Binary classification setup:
    - Class 0 (C1 - minority): DB (Database)
    - Class 1 (C2 - majority): All others (Agents, AI, IR, ML, HCI)
    
    Args:
        root: Root directory containing CiteSeer files
        alpha: Fuzzification parameter (default: 0.4)
        seed: Random seed
        
    Returns:
        data: PyG Data object
        fuzzy_labels: Fuzzy membership labels [num_nodes, 2]
        crisp_labels: Binary labels (0: DB, 1: Others)
    """
    # Import here to avoid circular dependency
    import sys
    import os as os_module
    sys.path.insert(0, os_module.path.join(os_module.path.dirname(__file__), '..'))
    from src.fuzzification import DatasetFuzzifier
    
    # Normalize paths
    root = os.path.normpath(root)
    
    # Expected files
    content_file = os.path.join(root, 'citeseer.content')
    cites_file = os.path.join(root, 'citeseer.cites')
    
    # Verify files exist
    if not os.path.exists(content_file):
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"❌ CITESEER DATASET NOT FOUND!\n"
            f"{'='*80}\n"
            f"Expected directory: {os.path.abspath(root)}\n\n"
            f"Required files:\n"
            f"  ✓ citeseer.content  (paper_id, words..., category)\n"
            f"  ✓ citeseer.cites    (cited_id, citing_id)\n\n"
            f"Download from:\n"
            f"https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip\n"
            f"{'='*80}\n"
        )
    
    print(f"✓ Loading CiteSeer from: {os.path.abspath(root)}")
    
    # Load content file
    # Format: <paper_id> <word_1> <word_2> ... <word_3703> <class_label>
    content_data = []
    with open(content_file, 'r', encoding='latin1', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')  # Try tab first
            if len(parts) == 1:  # If no tabs, try space
                parts = line.strip().split()
            if len(parts) >= 3:  # At least paper_id + 1 feature + label
                content_data.append(parts)
    
    print(f"  Loaded {len(content_data)} papers from content file")
    
    # Parse content
    paper_ids = [row[0] for row in content_data]
    labels_str = [row[-1] for row in content_data]
    
    # Features are all columns except first (paper_id) and last (label)
    features = []
    for row in content_data:
        try:
            feat = [int(val) for val in row[1:-1]]
            features.append(feat)
        except ValueError:
            # Skip malformed rows
            continue
    
    # Map paper IDs to indices
    paper_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    num_nodes = len(paper_ids)
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32)
    
    print(f"  Features shape: {x.shape}")
    
    # Map categories to binary labels
    # DB (minority) = 0, Others (majority) = 1
    category_mapping = {
        'DB': 0,      # Database - minority
        'Agents': 1,  # All others - majority
        'AI': 1,
        'IR': 1,
        'ML': 1,
        'HCI': 1
    }
    
    crisp_labels = torch.tensor([category_mapping.get(label, 1) for label in labels_str], dtype=torch.long)
    
    # Load citations (edges)
    edge_list = []
    with open(cites_file, 'r', encoding='latin1', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')  # Try tab first
            if len(parts) == 1:  # If no tabs, try space
                parts = line.strip().split()
            if len(parts) == 2:
                cited, citing = parts
                # Only include edges where both nodes exist
                if cited in paper_to_idx and citing in paper_to_idx:
                    cited_idx = paper_to_idx[cited]
                    citing_idx = paper_to_idx[citing]
                    edge_list.append([citing_idx, cited_idx])  # citing -> cited
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Count classes
    c1_count = (crisp_labels == 0).sum().item()  # DB
    c2_count = (crisp_labels == 1).sum().item()  # Others
    imbalance_ratio = c1_count / c2_count if c2_count > 0 else 0
    
    print("CiteSeer dataset loaded:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Features: {x.shape[1]}")
    print(f"  C1 (DB - minority): {c1_count} nodes")
    print(f"  C2 (Others - majority): {c2_count} nodes")
    print(f"  Imbalance ratio: {imbalance_ratio:.3f}")
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=crisp_labels)
    
    # Fuzzify labels
    fuzzifier = DatasetFuzzifier(
        alpha=alpha,
        min_membership=0.7,  # High confidence threshold
        seed=seed
    )
    
    fuzzy_labels = fuzzifier.fuzzify(data, crisp_labels)
    
    # Validate fuzzification
    high_conf_mask = fuzzy_labels.max(dim=1)[0] >= 0.7
    high_conf_count = high_conf_mask.sum().item()
    
    print(f"  High confidence nodes (μ >= 0.7): {high_conf_count}/{num_nodes}")
    
    return data, fuzzy_labels, crisp_labels


if __name__ == "__main__":
    # Test the loader
    data, fuzzy_labels, crisp_labels = load_citeseer()
    print("\n✓ CiteSeer loaded successfully!")
    print(f"  Data object: {data}")
    print(f"  Fuzzy labels shape: {fuzzy_labels.shape}")
    print(f"  Crisp labels shape: {crisp_labels.shape}")
