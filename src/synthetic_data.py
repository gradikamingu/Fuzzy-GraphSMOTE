"""
Générateur de datasets synthétiques pour Fuzzy-GraphSMOTE.
Utilise le modèle Barabási-Albert pour générer des graphes scale-free.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Tuple, List
import os


class SyntheticDatasetGenerator:
    """
    Génère des datasets synthétiques avec le modèle Barabási-Albert.
    
    Tailles: 100, 1,000, 10,000, 100,000, 1,000,000 nœuds
    Features: 50-dim Gaussiennes conditionnées par classe
    Labels: Fuzzy (probabiliste ou possibiliste)
    """
    
    def __init__(
        self,
        feature_dim: int = 50,
        m_edges: int = 5,
        seed: int = 42
    ):
        """
        Args:
            feature_dim: Dimension des features (défaut: 50)
            m_edges: Nombre d'arêtes ajoutées par nouveau nœud (défaut: 5)
            seed: Graine aléatoire
        """
        self.feature_dim = feature_dim
        self.m_edges = m_edges
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_graph(
        self,
        num_nodes: int,
        balanced: bool = True,
        framework: str = 'probabilistic'
    ) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Génère un graphe synthétique.
        
        Args:
            num_nodes: Nombre de nœuds (100, 1000, 10000, 100000, 1000000)
            balanced: Si True, classes équilibrées (50/50), sinon imbalanced (30/70)
            framework: 'probabilistic' ou 'possibilistic'
            
        Returns:
            data: PyG Data object
            fuzzy_labels: Fuzzy memberships [num_nodes, 2]
            crisp_labels: Crisp labels [num_nodes]
        """
        print(f"\nGénération d'un graphe synthétique...")
        print(f"  Nœuds: {num_nodes:,}")
        print(f"  m={self.m_edges} arêtes par nœud")
        print(f"  Balanced: {balanced}")
        print(f"  Framework: {framework}")
        
        # 1. Générer le graphe Barabási-Albert
        print("  [1/4] Génération du graphe BA...")
        G = nx.barabasi_albert_graph(num_nodes, self.m_edges, seed=self.seed)
        
        # Convertir en PyG Data
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        # Rendre non-orienté
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        print(f"        ✓ Graphe créé: {num_nodes} nœuds, {edge_index.shape[1]} arêtes")
        print(f"        Degré moyen: {edge_index.shape[1] / num_nodes:.2f}")
        
        # 2. Générer les centroids de classes
        print("  [2/4] Génération des centroids...")
        mu_1, mu_2 = self._generate_class_centroids()
        
        print(f"        ✓ Centroids générés")
        print(f"        Séparation: {np.linalg.norm(mu_1 - mu_2):.2f}")
        
        # 3. Assigner les classes
        print("  [3/4] Attribution des classes...")
        if balanced:
            # 50% C1, 50% C2
            num_c1 = num_nodes // 2
        else:
            # 30% C1 (minorité), 70% C2 (majorité)
            num_c1 = int(num_nodes * 0.3)
        
        num_c2 = num_nodes - num_c1
        
        crisp_labels = torch.cat([
            torch.zeros(num_c1, dtype=torch.long),
            torch.ones(num_c2, dtype=torch.long)
        ])
        
        # Permuter aléatoirement
        perm = torch.randperm(num_nodes)
        crisp_labels = crisp_labels[perm]
        
        print(f"        ✓ Classes attribuées")
        print(f"        C1: {num_c1} nœuds ({num_c1/num_nodes*100:.1f}%)")
        print(f"        C2: {num_c2} nœuds ({num_c2/num_nodes*100:.1f}%)")
        
        # 4. Générer les features
        print("  [4/4] Génération des features...")
        X = self._generate_features(crisp_labels, mu_1, mu_2)
        
        print(f"        ✓ Features générées: {X.shape}")
        
        # 5. Créer le Data object
        data = Data(x=X, edge_index=edge_index, y=crisp_labels)
        
        # 6. Fuzzifier les labels
        print("  Fuzzification des labels...")
        fuzzy_labels = self._fuzzify_labels(
            data, crisp_labels, framework, balanced
        )
        
        print(f"✓ Dataset synthétique créé!")
        
        return data, fuzzy_labels, crisp_labels
    
    def _generate_class_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère les centroids de classes avec contrainte de séparation.
        
        Contrainte: ||μ_1 - μ_2||_2 >= 3.0
        """
        max_attempts = 100
        
        for _ in range(max_attempts):
            mu_1 = np.random.uniform(-2, 2, self.feature_dim)
            mu_2 = np.random.uniform(-2, 2, self.feature_dim)
            
            separation = np.linalg.norm(mu_1 - mu_2)
            
            if separation >= 3.0:
                return mu_1, mu_2
        
        # Si échec, forcer la séparation
        mu_1 = np.random.uniform(-2, 2, self.feature_dim)
        direction = np.random.randn(self.feature_dim)
        direction = direction / np.linalg.norm(direction)
        mu_2 = mu_1 + 3.5 * direction
        
        # Clipper dans [-2, 2]
        mu_2 = np.clip(mu_2, -2, 2)
        
        return mu_1, mu_2
    
    def _generate_features(
        self,
        crisp_labels: torch.Tensor,
        mu_1: np.ndarray,
        mu_2: np.ndarray
    ) -> torch.Tensor:
        """
        Génère les features à partir de Gaussiennes conditionnées par classe.
        
        x_i ~ N(μ_k, 0.5 * I_50) où k est la classe dominante de i
        """
        num_nodes = len(crisp_labels)
        X = np.zeros((num_nodes, self.feature_dim))
        
        # Covariance: 0.5 * I
        cov = 0.5 * np.eye(self.feature_dim)
        
        for i in range(num_nodes):
            if crisp_labels[i] == 0:
                # Classe C1
                X[i] = np.random.multivariate_normal(mu_1, cov)
            else:
                # Classe C2
                X[i] = np.random.multivariate_normal(mu_2, cov)
        
        return torch.tensor(X, dtype=torch.float32)
    
    def _fuzzify_labels(
        self,
        data: Data,
        crisp_labels: torch.Tensor,
        framework: str,
        balanced: bool
    ) -> torch.Tensor:
        """
        Fuzzifie les labels selon le framework spécifié.
        
        Utilise la même stratégie que pour les datasets réels:
        - Framework probabiliste: basé sur voisinage
        - Framework possibiliste: basé sur distance aux centroids
        - 5% de nœuds ambigus
        - μ_dominant >= 0.6 pour les autres
        """
        from fuzzification import DatasetFuzzifier
        
        fuzzifier = DatasetFuzzifier(
            framework=framework,
            alpha=0.4,
            ambiguous_ratio=0.05,
            min_membership=0.6,
            seed=self.seed
        )
        
        fuzzy_labels = fuzzifier.fuzzify(data, crisp_labels)
        
        return fuzzy_labels
    
    def generate_all_sizes(
        self,
        balanced: bool = True,
        framework: str = 'probabilistic',
        num_instances: int = 5,
        save_dir: str = './data/synthetic'
    ) -> List[Tuple[Data, torch.Tensor, torch.Tensor]]:
        """
        Génère tous les datasets de différentes tailles.
        
        Tailles: 100, 1,000, 10,000, 100,000, 1,000,000
        
        Args:
            balanced: Classes équilibrées ou non
            framework: Framework fuzzy
            num_instances: Nombre d'instances par taille (défaut: 5)
            save_dir: Répertoire de sauvegarde
            
        Returns:
            Liste de datasets pour chaque taille
        """
        sizes = [100, 1000, 10000, 100000, 1000000]
        # sizes_log = [100 * 10**k for k in range(5)]
        
        os.makedirs(save_dir, exist_ok=True)
        
        all_datasets = {}
        
        for size in sizes:
            print(f"\n{'='*80}")
            print(f"Taille: {size:,} nœuds")
            print(f"{'='*80}")
            
            datasets_for_size = []
            
            for instance in range(num_instances):
                print(f"\n--- Instance {instance + 1}/{num_instances} ---")
                
                # Changer la seed pour chaque instance
                self.seed = 42 + instance
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                
                data, fuzzy_labels, crisp_labels = self.generate_graph(
                    num_nodes=size,
                    balanced=balanced,
                    framework=framework
                )
                
                datasets_for_size.append((data, fuzzy_labels, crisp_labels))
                
                # Sauvegarder
                balance_str = 'balanced' if balanced else 'imbalanced'
                filename = f'synthetic_{size}_{balance_str}_{framework}_inst{instance}.pt'
                filepath = os.path.join(save_dir, filename)
                
                torch.save({
                    'data': data,
                    'fuzzy_labels': fuzzy_labels,
                    'crisp_labels': crisp_labels,
                    'params': {
                        'num_nodes': size,
                        'balanced': balanced,
                        'framework': framework,
                        'feature_dim': self.feature_dim,
                        'm_edges': self.m_edges,
                        'instance': instance,
                        'seed': self.seed
                    }
                }, filepath)
                
                print(f"  ✓ Sauvegardé: {filepath}")
            
            all_datasets[size] = datasets_for_size
        
        print(f"\n{'='*80}")
        print(f"✓ Tous les datasets générés et sauvegardés dans: {save_dir}")
        print(f"{'='*80}")
        
        return all_datasets


# Fonctions utilitaires

def load_synthetic_dataset(
    num_nodes: int,
    balanced: bool = True,
    framework: str = 'probabilistic',
    instance: int = 0,
    data_dir: str = './data/synthetic'
) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Charge un dataset synthétique sauvegardé.
    """
    balance_str = 'balanced' if balanced else 'imbalanced'
    filename = f'synthetic_{num_nodes}_{balance_str}_{framework}_inst{instance}.pt'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset non trouvé: {filepath}")
    
    checkpoint = torch.load(filepath)
    
    return checkpoint['data'], checkpoint['fuzzy_labels'], checkpoint['crisp_labels']


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*80)
    print("GÉNÉRATEUR DE DATASETS SYNTHÉTIQUES")
    print("="*80)
    
    generator = SyntheticDatasetGenerator(feature_dim=50, m_edges=5, seed=42)
    
    # Test: générer un petit graphe
    print("\n[Test] Génération d'un graphe de test (1,000 nœuds)...")
    data, fuzzy_labels, crisp_labels = generator.generate_graph(
        num_nodes=1000,
        balanced=False,  # Imbalanced
        framework='probabilistic'
    )
    
    print("\nStatistiques du graphe généré:")
    print(f"  Nœuds: {data.num_nodes}")
    print(f"  Arêtes: {data.num_edges}")
    print(f"  Features: {data.x.shape}")
    print(f"  C1 (minorité): {(crisp_labels == 0).sum()} nœuds")
    print(f"  C2 (majorité): {(crisp_labels == 1).sum()} nœuds")
    print(f"  Fuzzy labels: {fuzzy_labels.shape}")
    print(f"  Nœuds avec μ >= 0.6: {(fuzzy_labels.max(dim=1)[0] >= 0.6).sum()}")
    
    # Générer tous les datasets (commenté par défaut car long)
    # print("\n[Production] Génération de tous les datasets...")
    # all_datasets = generator.generate_all_sizes(
    #     balanced=True,
    #     framework='probabilistic',
    #     num_instances=5
    # )
    
    print("\n✓ Test terminé!")
    print("\nPour générer tous les datasets, décommenter le code ci-dessus.")
    print("Cela générera 5 instances de chaque taille (100 à 1,000,000 nœuds).")
