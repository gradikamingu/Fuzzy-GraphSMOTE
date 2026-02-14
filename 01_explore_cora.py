"""
Script pour télécharger Cora et afficher les nœuds sous forme de tableau.
Usage: python explore_cora.py
"""

import sys
sys.path.append('./src')

import torch
import pandas as pd
import numpy as np
from data_loader import load_cora
from torch_geometric.datasets import Planetoid

def display_cora_nodes_table(save_csv=True):
    """
    Télécharge Cora et affiche les nœuds avec leurs classes.
    """
    print("="*80)
    print("EXPLORATION DU DATASET CORA")
    print("="*80)
    
    # 1. Charger le dataset original Cora (7 classes)
    print("\n[1] Téléchargement du dataset Cora original...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    print(f"✓ Dataset téléchargé!")
    print(f"  Nœuds: {data.num_nodes}")
    print(f"  Arêtes: {data.num_edges}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Classes originales: {data.y.max().item() + 1}")
    
    # Noms des classes originales
    class_names = [
        'Theory',              # 0
        'Reinforcement_Learning',  # 1
        'Rule_Learning',       # 2
        'Neural_Networks',     # 3
        'Case_Based',          # 4
        'Genetic_Algorithms',  # 5
        'Probabilistic_Methods'  # 6
    ]
    
    # 2. Créer les meta-classes
    print("\n[2] Création des meta-classes...")
    
    # Mapping: classe originale -> meta-classe
    # C1 (Logic/Theory): Theory(0), Rule_Learning(2), Case_Based(4)
    # C2 (Practical/Applied): RL(1), Neural_Networks(3), Genetic_Algorithms(5), Probabilistic_Methods(6)
    
    meta_class_mapping = {
        0: 'C1_Logic_Theory',      # Theory
        2: 'C1_Logic_Theory',      # Rule_Learning
        4: 'C1_Logic_Theory',      # Case_Based
        1: 'C2_Practical_Applied', # Reinforcement_Learning
        3: 'C2_Practical_Applied', # Neural_Networks
        5: 'C2_Practical_Applied', # Genetic_Algorithms
        6: 'C2_Practical_Applied'  # Probabilistic_Methods
    }
    
    meta_class_binary_mapping = {
        0: 0,  # Theory -> C1
        2: 0,  # Rule_Learning -> C1
        4: 0,  # Case_Based -> C1
        1: 1,  # Reinforcement_Learning -> C2
        3: 1,  # Neural_Networks -> C2
        5: 1,  # Genetic_Algorithms -> C2
        6: 1   # Probabilistic_Methods -> C2
    }
    
    # 3. Créer le DataFrame
    print("\n[3] Création du tableau des nœuds...")
    
    node_data = []
    for i in range(data.num_nodes):
        original_class_id = data.y[i].item()
        original_class_name = class_names[original_class_id]
        meta_class_name = meta_class_mapping[original_class_id]
        meta_class_binary = meta_class_binary_mapping[original_class_id]
        
        # Degré du nœud (nombre de voisins)
        degree = (data.edge_index[0] == i).sum().item()
        
        node_data.append({
            'Node_ID': i,
            'Original_Class_ID': original_class_id,
            'Original_Class_Name': original_class_name,
            'Meta_Class': meta_class_name,
            'Meta_Class_Binary': meta_class_binary,
            'Degree': degree
        })
    
    df = pd.DataFrame(node_data)
    
    # 4. Afficher des statistiques
    print("\n[4] Statistiques des classes:")
    print("\n--- Classes Originales (7 classes) ---")
    print(df['Original_Class_Name'].value_counts().sort_index())
    
    print("\n--- Meta-Classes (2 classes) ---")
    print(df['Meta_Class'].value_counts())
    
    print("\n--- Distribution Binaire ---")
    c1_count = (df['Meta_Class_Binary'] == 0).sum()
    c2_count = (df['Meta_Class_Binary'] == 1).sum()
    print(f"C1 (Logic/Theory) - MINORITÉ: {c1_count} nœuds ({c1_count/len(df)*100:.1f}%)")
    print(f"C2 (Practical/Applied) - MAJORITÉ: {c2_count} nœuds ({c2_count/len(df)*100:.1f}%)")
    print(f"Imbalance Ratio: {c1_count/c2_count:.3f}")
    
    # 5. Afficher les premières lignes
    print("\n[5] Aperçu des données (20 premières lignes):")
    print("="*100)
    print(df.head(20).to_string(index=False))
    print("="*100)
    
    # 6. Afficher quelques exemples de chaque classe
    print("\n[6] Exemples par classe originale:")
    for class_id in range(7):
        class_name = class_names[class_id]
        samples = df[df['Original_Class_ID'] == class_id].head(3)
        print(f"\n{class_name} (ID={class_id}):")
        print(samples[['Node_ID', 'Original_Class_Name', 'Meta_Class', 'Degree']].to_string(index=False))
    
    # 7. Sauvegarder en CSV
    if save_csv:
        csv_file = './data/cora_nodes_table.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Tableau sauvegardé dans: {csv_file}")
    
    # 8. Statistiques des degrés par classe
    print("\n[7] Statistiques des degrés par meta-classe:")
    print("\nC1 (Logic/Theory):")
    c1_degrees = df[df['Meta_Class_Binary'] == 0]['Degree']
    print(f"  Moyenne: {c1_degrees.mean():.2f}")
    print(f"  Médiane: {c1_degrees.median():.0f}")
    print(f"  Min: {c1_degrees.min()}, Max: {c1_degrees.max()}")
    
    print("\nC2 (Practical/Applied):")
    c2_degrees = df[df['Meta_Class_Binary'] == 1]['Degree']
    print(f"  Moyenne: {c2_degrees.mean():.2f}")
    print(f"  Médiane: {c2_degrees.median():.0f}")
    print(f"  Min: {c2_degrees.min()}, Max: {c2_degrees.max()}")
    
    return df


def display_cora_with_fuzzy_labels():
    """
    Affiche Cora avec les labels flous.
    """
    print("\n" + "="*80)
    print("CORA AVEC LABELS FLOUS")
    print("="*80)
    
    # Charger avec fuzzification
    print("\n[1] Chargement avec fuzzification probabiliste...")
    data, fuzzy_labels, crisp_labels = load_cora(
        framework='probabilistic',
        alpha=0.4,
        seed=42
    )
    
    # Créer DataFrame
    node_data = []
    for i in range(min(50, data.num_nodes)):  # Afficher 50 premiers
        node_data.append({
            'Node_ID': i,
            'Crisp_Class': crisp_labels[i].item(),
            'Fuzzy_C1': f"{fuzzy_labels[i, 0].item():.4f}",
            'Fuzzy_C2': f"{fuzzy_labels[i, 1].item():.4f}",
            'Sum': f"{fuzzy_labels[i].sum().item():.4f}",
            'Dominant': 'C1' if fuzzy_labels[i, 0] > fuzzy_labels[i, 1] else 'C2',
            'Confidence': f"{fuzzy_labels[i].max().item():.4f}"
        })
    
    df_fuzzy = pd.DataFrame(node_data)
    
    print("\n[2] Aperçu des labels flous (50 premiers nœuds):")
    print("="*100)
    print(df_fuzzy.to_string(index=False))
    print("="*100)
    
    # Statistiques
    print("\n[3] Statistiques des fuzzy memberships:")
    print(f"Nœuds avec confiance >= 0.6: {(fuzzy_labels.max(dim=1)[0] >= 0.6).sum()}/{data.num_nodes}")
    print(f"Nœuds avec confiance < 0.6 (ambigus): {(fuzzy_labels.max(dim=1)[0] < 0.6).sum()}/{data.num_nodes}")
    
    # Identifier les nœuds ambigus
    ambiguous_mask = fuzzy_labels.max(dim=1)[0] < 0.6
    if ambiguous_mask.sum() > 0:
        print(f"\nNœuds ambigus (μ_max < 0.6):")
        ambiguous_indices = torch.where(ambiguous_mask)[0][:10]  # 10 premiers
        for idx in ambiguous_indices:
            print(f"  Node {idx.item()}: μ_C1={fuzzy_labels[idx, 0]:.4f}, μ_C2={fuzzy_labels[idx, 1]:.4f}")
    
    # Sauvegarder
    csv_file = './data/cora_fuzzy_labels.csv'
    df_full = pd.DataFrame({
        'Node_ID': range(data.num_nodes),
        'Crisp_Class': crisp_labels.numpy(),
        'Fuzzy_C1': fuzzy_labels[:, 0].numpy(),
        'Fuzzy_C2': fuzzy_labels[:, 1].numpy(),
    })
    df_full.to_csv(csv_file, index=False)
    print(f"\n✓ Labels flous sauvegardés dans: {csv_file}")


if __name__ == "__main__":
    print("\n" + "🔍 EXPLORATION DU DATASET CORA\n")
    
    # 1. Afficher le tableau des nœuds avec classes
    df = display_cora_nodes_table(save_csv=True)
    
    # 2. Afficher avec labels flous
    display_cora_with_fuzzy_labels()
    
    print("\n" + "="*80)
    print("✓ Exploration terminée!")
    print("="*80)
    print("\nFichiers créés:")
    print("  - ./data/cora_nodes_table.csv (nœuds avec classes)")
    print("  - ./data/cora_fuzzy_labels.csv (labels flous)")
    print("\nVous pouvez maintenant:")
    print("  1. Ouvrir les CSV dans Excel/Google Sheets")
    print("  2. Utiliser le notebook analysis.ipynb pour visualiser")
    print("  3. Lancer des expériences avec run_experiments.py")
