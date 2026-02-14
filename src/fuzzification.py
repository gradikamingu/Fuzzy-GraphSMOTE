"""
Fuzzification module for converting crisp labels to fuzzy memberships.
Implements both probabilistic and possibilistic frameworks.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch_geometric.data import Data


class DatasetFuzzifier:
    """
    Converts crisp labels to fuzzy memberships following probabilistic 
    or possibilistic frameworks.
    """
    
    def __init__(
        self,
        framework: str = 'probabilistic',
        alpha: float = 0.4,
        ambiguous_ratio: float = 0.05,
        min_membership: float = 0.7,  # CHANGÉ: 0.6 → 0.7
        seed: int = 42
    ):
        """
        Args:
            framework: 'probabilistic' or 'possibilistic'
            alpha: Control parameter for epsilon calculation (0 < alpha <= 0.4)
            ambiguous_ratio: Proportion of nodes with ambiguous memberships (default 5%)
            min_membership: Minimum membership for dominant class (default 0.7)
            seed: Random seed for reproducibility
        """
        assert framework in ['probabilistic', 'possibilistic'], \
            "framework must be 'probabilistic' or 'possibilistic'"
        assert 0 < alpha <= 0.4, "alpha must be in (0, 0.4]"
        assert 0 < ambiguous_ratio < 1, "ambiguous_ratio must be in (0, 1)"
        assert 0.5 < min_membership < 1, "min_membership must be in (0.5, 1)"
        
        self.framework = framework
        self.alpha = alpha
        self.ambiguous_ratio = ambiguous_ratio
        self.min_membership = min_membership
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def fuzzify(
        self, 
        data: Data,
        y_crisp: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert crisp labels to fuzzy memberships.
        
        Args:
            data: PyG Data object containing edge_index and x
            y_crisp: Crisp labels [num_nodes]
            
        Returns:
            fuzzy_labels: Fuzzy membership matrix [num_nodes, num_classes]
        """
        num_nodes = y_crisp.shape[0]
        num_classes = int(y_crisp.max().item()) + 1
        
        if self.framework == 'probabilistic':
            fuzzy_labels = self._probabilistic_fuzzification(
                data, y_crisp, num_nodes, num_classes
            )
        else:  # possibilistic
            fuzzy_labels = self._possibilistic_fuzzification(
                data, y_crisp, num_nodes, num_classes
            )
        
        # Inject controlled ambiguity for realistic uncertainty
        fuzzy_labels = self._inject_ambiguity(
            data, y_crisp, fuzzy_labels, num_classes
        )
        
        return fuzzy_labels
    
    def _probabilistic_fuzzification(
        self,
        data: Data,
        y_crisp: torch.Tensor,
        num_nodes: int,
        num_classes: int
    ) -> torch.Tensor:
        """
        Probabilistic fuzzification based on neighborhood composition.
        
        Constraint: sum of memberships = 1 for each node
        Formula: μ_Ci(v) = 1 - ε(v) if i = y(v), else ε(v)
        where ε(v) = α * |N(v) ∩ Cj| / |N(v)|
        """
        edge_index = data.edge_index
        fuzzy_labels = torch.zeros(num_nodes, num_classes)
        
        # Build adjacency list for efficient neighbor queries
        neighbors = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            neighbors[src].append(dst)
        
        for v in range(num_nodes):
            true_class = y_crisp[v].item()
            neighbor_list = neighbors[v]
            
            if len(neighbor_list) == 0:
                # Isolated node: assign full membership to true class
                fuzzy_labels[v, true_class] = 1.0
                continue
            
            # Count neighbors in each class
            neighbor_classes = [y_crisp[n].item() for n in neighbor_list]
            neighbor_counts = np.bincount(neighbor_classes, minlength=num_classes)
            
            # Calculate epsilon based on neighbors NOT in true class
            total_neighbors = len(neighbor_list)
            neighbors_other_class = total_neighbors - neighbor_counts[true_class]
            epsilon = self.alpha * (neighbors_other_class / total_neighbors)
            epsilon = min(epsilon, 0.4)  # Cap at 0.4
            
            # Assign memberships
            fuzzy_labels[v, true_class] = 1.0 - epsilon
            
            # Distribute epsilon among other classes proportionally
            if num_classes == 2:
                other_class = 1 - true_class
                fuzzy_labels[v, other_class] = epsilon
            else:
                # Multi-class: distribute proportionally to neighbor counts
                other_classes_count = neighbor_counts.copy()
                other_classes_count[true_class] = 0
                total_other = other_classes_count.sum()
                
                if total_other > 0:
                    for c in range(num_classes):
                        if c != true_class:
                            fuzzy_labels[v, c] = epsilon * (other_classes_count[c] / total_other)
                else:
                    # No neighbors in other classes: distribute uniformly
                    fuzzy_labels[v, :] = epsilon / (num_classes - 1)
                    fuzzy_labels[v, true_class] = 1.0 - epsilon
            
            # Normalize to ensure sum = 1 (probabilistic constraint)
            fuzzy_labels[v] = fuzzy_labels[v] / fuzzy_labels[v].sum()
        
        return fuzzy_labels
    
    def _possibilistic_fuzzification(
        self,
        data: Data,
        y_crisp: torch.Tensor,
        num_nodes: int,
        num_classes: int
    ) -> torch.Tensor:
        """
        Possibilistic fuzzification based on distance to class centroids.
        
        No constraint on sum of memberships.
        Formula: μ_Ci(v) = exp(-||x_v - p_Ci||² / (2σ_i²))
        """
        x = data.x  # Node features
        fuzzy_labels = torch.zeros(num_nodes, num_classes)
        
        # Compute class centroids
        centroids = []
        sigmas = []
        
        for c in range(num_classes):
            class_mask = (y_crisp == c)
            class_features = x[class_mask]
            
            if class_features.shape[0] == 0:
                # Empty class (shouldn't happen in practice)
                centroids.append(torch.zeros(x.shape[1]))
                sigmas.append(1.0)
                continue
            
            # Centroid: mean of features in class c
            centroid = class_features.mean(dim=0)
            centroids.append(centroid)
            
            # Sigma: average intra-class Euclidean distance
            distances = torch.norm(class_features - centroid, dim=1)
            sigma = distances.mean().item()
            if sigma < 1e-6:  # Avoid division by zero
                sigma = 1.0
            sigmas.append(sigma)
        
        # Compute memberships based on distance to centroids
        for v in range(num_nodes):
            for c in range(num_classes):
                dist_sq = torch.norm(x[v] - centroids[c]) ** 2
                membership = np.exp(-dist_sq.item() / (2 * sigmas[c] ** 2))
                fuzzy_labels[v, c] = membership
        
        # Ensure interpretability: dominant membership >= 0.6
        for v in range(num_nodes):
            max_membership = fuzzy_labels[v].max().item()
            if max_membership < self.min_membership:
                # Rescale proportionally
                scale_factor = self.min_membership / max_membership
                fuzzy_labels[v] = fuzzy_labels[v] * scale_factor
                # Re-normalize to keep the dominant at min_membership
                fuzzy_labels[v] = fuzzy_labels[v] / fuzzy_labels[v].max() * self.min_membership
        
        return fuzzy_labels
    
    def _inject_ambiguity(
        self,
        data: Data,
        y_crisp: torch.Tensor,
        fuzzy_labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Inject controlled ambiguity for realistic uncertainty.
        
        Select 5% of nodes with balanced neighborhoods and assign
        ambiguous memberships: [0.45, 0.55] for probabilistic,
        [0.4, 0.6] independently for possibilistic.
        """
        edge_index = data.edge_index
        num_nodes = fuzzy_labels.shape[0]
        
        # Build adjacency list
        neighbors = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            neighbors[src].append(dst)
        
        # Find nodes with balanced neighborhoods
        balanced_nodes = []
        
        for v in range(num_nodes):
            neighbor_list = neighbors[v]
            if len(neighbor_list) < 2:  # Need at least 2 neighbors
                continue
            
            # Count neighbors in each class
            neighbor_classes = [y_crisp[n].item() for n in neighbor_list]
            neighbor_counts = np.bincount(neighbor_classes, minlength=num_classes)
            
            if num_classes == 2:
                # Binary case: check if neighbors are nearly balanced
                diff = abs(neighbor_counts[0] - neighbor_counts[1])
                threshold = 0.1 * len(neighbor_list)
                if diff <= threshold:
                    balanced_nodes.append(v)
            else:
                # Multi-class: check if top two classes are nearly balanced
                top_two = np.sort(neighbor_counts)[-2:]
                if len(top_two) == 2:
                    diff = abs(top_two[1] - top_two[0])
                    threshold = 0.1 * len(neighbor_list)
                    if diff <= threshold:
                        balanced_nodes.append(v)
        
        # Randomly sample ambiguous_ratio of total dataset from balanced nodes
        num_ambiguous = int(self.ambiguous_ratio * num_nodes)
        if len(balanced_nodes) > num_ambiguous:
            ambiguous_indices = np.random.choice(
                balanced_nodes, size=num_ambiguous, replace=False
            )
        else:
            ambiguous_indices = balanced_nodes
        
        # Assign ambiguous memberships
        for v in ambiguous_indices:
            if self.framework == 'probabilistic':
                if num_classes == 2:
                    # [0.45, 0.55] for binary
                    mu1 = np.random.uniform(0.45, 0.55)
                    fuzzy_labels[v, 0] = mu1
                    fuzzy_labels[v, 1] = 1.0 - mu1
                else:
                    # Multi-class: distribute more uniformly
                    memberships = np.random.dirichlet(np.ones(num_classes) * 2)
                    fuzzy_labels[v] = torch.tensor(memberships, dtype=torch.float32)
            else:  # possibilistic
                if num_classes == 2:
                    # [0.4, 0.6] independently for binary
                    fuzzy_labels[v, 0] = np.random.uniform(0.4, 0.6)
                    fuzzy_labels[v, 1] = np.random.uniform(0.4, 0.6)
                else:
                    # Multi-class: each membership in [0.3, 0.6]
                    for c in range(num_classes):
                        fuzzy_labels[v, c] = np.random.uniform(0.3, 0.6)
        
        return fuzzy_labels


def create_cora_metaclasses(y_original: torch.Tensor) -> torch.Tensor:
    """
    Convert Cora's 7 classes into 2 meta-classes:
    - C1 (Logic and Theoretical): Theory (0), Rule Learning (1), Case Based (2)
    - C2 (Practical and Applied): Neural Networks (3), Probabilistic (4), 
                                  Genetic Algorithms (5), Reinforcement Learning (6)
    
    Args:
        y_original: Original labels [num_nodes] with values 0-6
        
    Returns:
        y_meta: Meta-class labels [num_nodes] with values 0-1
    """
    y_meta = torch.zeros_like(y_original)
    
    # C1: Classes 0, 1, 2 (Theory, Rule Learning, Case Based)
    c1_mask = (y_original == 0) | (y_original == 1) | (y_original == 2)
    
    # C2: Classes 3, 4, 5, 6 (Neural Networks, Probabilistic, Genetic, RL)
    c2_mask = (y_original == 3) | (y_original == 4) | (y_original == 5) | (y_original == 6)
    
    y_meta[c1_mask] = 0
    y_meta[c2_mask] = 1
    
    return y_meta


# Example usage
if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    
    # Load Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # Convert to meta-classes
    y_meta = create_cora_metaclasses(data.y)
    
    print(f"Original classes distribution: {torch.bincount(data.y)}")
    print(f"Meta-classes distribution: {torch.bincount(y_meta)}")
    print(f"C1 (minority): {(y_meta == 0).sum().item()} nodes")
    print(f"C2 (majority): {(y_meta == 1).sum().item()} nodes")
    print(f"Imbalance ratio: {(y_meta == 0).sum().item() / (y_meta == 1).sum().item():.3f}")
    
    # Test probabilistic fuzzification
    print("\n=== Probabilistic Fuzzification ===")
    fuzzifier_prob = DatasetFuzzifier(framework='probabilistic', alpha=0.4)
    fuzzy_prob = fuzzifier_prob.fuzzify(data, y_meta)
    
    print(f"Fuzzy labels shape: {fuzzy_prob.shape}")
    print(f"Sample fuzzy labels (first 5 nodes):\n{fuzzy_prob[:5]}")
    print(f"Sum of memberships (should be ~1): {fuzzy_prob[:5].sum(dim=1)}")
    print(f"Nodes with membership >= 0.6 in dominant class: "
          f"{(fuzzy_prob.max(dim=1)[0] >= 0.6).sum().item()} / {fuzzy_prob.shape[0]}")
    
    # Test possibilistic fuzzification
    print("\n=== Possibilistic Fuzzification ===")
    fuzzifier_poss = DatasetFuzzifier(framework='possibilistic', alpha=0.4)
    fuzzy_poss = fuzzifier_poss.fuzzify(data, y_meta)
    
    print(f"Fuzzy labels shape: {fuzzy_poss.shape}")
    print(f"Sample fuzzy labels (first 5 nodes):\n{fuzzy_poss[:5]}")
    print(f"Sum of memberships (no constraint): {fuzzy_poss[:5].sum(dim=1)}")
    print(f"Nodes with membership >= 0.6 in dominant class: "
          f"{(fuzzy_poss.max(dim=1)[0] >= 0.6).sum().item()} / {fuzzy_poss.shape[0]}")
