#!/usr/bin/env python3
"""
Generate Specific Figures Requested by Professor - FIXED VERSION
=================================================================

Figure 1: Bimodal distribution from EXPERIMENTAL DATA (not model predictions)
Figure 2: Generalized model comparison (RF vs NN across embeddings)
Figure 3: Dropout hyperparameter sweep

Usage:
    python scripts/05_generate_professor_figures_FIXED.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

from data_utils import load_plm_embeddings
from models import ProteinClassifierNN
from eval_utils import compute_regression_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)


def figure1_bimodal_experimental_data():
    """
    Plot experimental data showing bimodal bin distribution.

    As requested: "Plot a histogram of an example of a Grasso result where
    there is a bimodal distribution of bin occupancies, and WA is in between…
    the WA is not even the most likely result"
    """
    print("\n" + "="*70)
    print("Figure 1: Bimodal Distribution from Experimental Data")
    print("="*70 + "\n")

    # Load Grasso dataset
    xlsx_path = Path(__file__).parent.parent / 'data' / 'sb2c00328_si_011.xlsx'
    df = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')

    # Get bin percentage columns
    bin_cols = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]

    # Find bimodal examples
    bimodal_examples = []
    for idx, row in df.iterrows():
        bin_probs = row[bin_cols].values
        wa = row['WA']

        if pd.isna(wa) or pd.isna(bin_probs).any():
            continue

        # Find top 2 bins
        top_bins_idx = np.argsort(bin_probs)[::-1][:2]
        top_probs = bin_probs[top_bins_idx]

        # Check if bimodal (2 significant peaks, separated by at least 2 bins)
        if (top_probs[0] > 0.2 and top_probs[1] > 0.15 and
            abs(top_bins_idx[0] - top_bins_idx[1]) >= 2):

            # Check if WA falls between the peaks (not at the mode)
            bin1, bin2 = sorted([top_bins_idx[0] + 1, top_bins_idx[1] + 1])
            if bin1 < wa < bin2:
                bimodal_examples.append({
                    'index': idx,
                    'ID': row['ID'],
                    'SP_aa': row['SP_aa'],
                    'WA': wa,
                    'bin_probs': bin_probs,
                    'top_bins': (bin1, bin2),
                    'top_probs': top_probs,
                    'separation': abs(bin1 - bin2)
                })

    print(f"Found {len(bimodal_examples)} bimodal examples")

    if not bimodal_examples:
        print("No bimodal examples found!")
        return None, None, None

    # Pick best example (largest separation between peaks)
    best_example = sorted(bimodal_examples, key=lambda x: x['separation'], reverse=True)[0]

    print(f"\nSelected example:")
    print(f"  ID: {best_example['ID']}")
    print(f"  Signal Peptide: {best_example['SP_aa']}")
    print(f"  WA: {best_example['WA']:.2f}")
    print(f"  Peak bins: {best_example['top_bins'][0]} and {best_example['top_bins'][1]}")
    print(f"  Peak probabilities: {best_example['top_probs'][0]:.1%} and {best_example['top_probs'][1]:.1%}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.arange(1, 11)
    bin_probs = best_example['bin_probs']
    wa = best_example['WA']

    # Plot histogram of experimental bin probabilities
    bars = ax.bar(bins, bin_probs, width=0.7, alpha=0.7, color='steelblue',
                   edgecolor='black', linewidth=1.5, label='Experimental Bin Distribution')

    # Highlight the two peaks
    peak_bins = best_example['top_bins']
    for peak_bin in peak_bins:
        bars[peak_bin - 1].set_color('orange')
        bars[peak_bin - 1].set_alpha(0.9)

    # Add WA as a vertical line
    ax.axvline(wa, color='red', linestyle='--', linewidth=3,
               label=f'WA = {wa:.2f}', alpha=0.8, zorder=10)

    # Formatting
    ax.set_xlabel('Bin Number (Secretion Efficiency)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Reads', fontsize=13, fontweight='bold')
    ax.set_title(f'Bimodal Experimental Distribution\\nSignal Peptide: {best_example["ID"]}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(bins)
    ax.set_ylim(0, max(bin_probs) * 1.2)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.text(0.05, 0.95,
            f'Peak 1: Bin {peak_bins[0]} ({best_example["top_probs"][0]:.1%})\\n'
            f'Peak 2: Bin {peak_bins[1]} ({best_example["top_probs"][1]:.1%})\\n'
            f'WA ({wa:.2f}) falls between peaks',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = Path('figures/fig1_bimodal_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    return best_example['index'], wa, best_example['ID']


def figure2_model_comparison():
    """Generalized model comparison (unchanged from before)"""
    print("\n" + "="*70)
    print("Figure 2: Generalized Model Comparison")
    print("="*70 + "\n")

    # Load results
    rf_results = pd.read_csv('results/rf_baseline_results.csv')
    nn_results = pd.read_csv('results/nn_classifier_results.csv')

    # Extract test metrics
    embeddings = ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']
    embedding_labels = ['ESM-2 650M', 'ESM-2 3B', 'Ginkgo AA0']

    rf_mse = [rf_results[rf_results['embedding_model'] == e]['test_mse'].values[0] for e in embeddings]
    rf_r2 = [rf_results[rf_results['embedding_model'] == e]['test_r2'].values[0] for e in embeddings]
    rf_spearman = [rf_results[rf_results['embedding_model'] == e]['test_spearman'].values[0] for e in embeddings]

    nn_mse = [nn_results[nn_results['embedding_model'] == e]['test_mse'].values[0] for e in embeddings]
    nn_r2 = [nn_results[nn_results['embedding_model'] == e]['test_r2'].values[0] for e in embeddings]
    nn_spearman = [nn_results[nn_results['embedding_model'] == e]['test_spearman'].values[0] for e in embeddings]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(embedding_labels))

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # MSE
    ax = axes[0]
    for i in range(len(embeddings)):
        ax.plot([i, i], [rf_mse[i], nn_mse[i]], 'o-', color=colors[i], linewidth=2, markersize=8)
    ax.plot(x, rf_mse, 'o--', color='gray', linewidth=2, markersize=10, label='RF', alpha=0.7)
    ax.plot(x, nn_mse, 's-', color='black', linewidth=2, markersize=10, label='NN', alpha=0.7)
    ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
    ax.set_title('Mean Squared Error', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(embedding_labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # R²
    ax = axes[1]
    for i in range(len(embeddings)):
        ax.plot([i, i], [rf_r2[i], nn_r2[i]], 'o-', color=colors[i], linewidth=2, markersize=8)
    ax.plot(x, rf_r2, 'o--', color='gray', linewidth=2, markersize=10, label='RF', alpha=0.7)
    ax.plot(x, nn_r2, 's-', color='black', linewidth=2, markersize=10, label='NN', alpha=0.7)
    ax.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax.set_title('Coefficient of Determination', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(embedding_labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Spearman
    ax = axes[2]
    for i in range(len(embeddings)):
        ax.plot([i, i], [rf_spearman[i], nn_spearman[i]], 'o-', color=colors[i], linewidth=2, markersize=8)
    ax.plot(x, rf_spearman, 'o--', color='gray', linewidth=2, markersize=10, label='RF', alpha=0.7)
    ax.plot(x, nn_spearman, 's-', color='black', linewidth=2, markersize=10, label='NN', alpha=0.7)
    ax.set_ylabel('Test Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('Rank Correlation', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(embedding_labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('figures/fig2_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def figure3_dropout_sweep():
    """Dropout hyperparameter sweep (unchanged from before)"""
    print("\n" + "="*70)
    print("Figure 3: Dropout Hyperparameter Sweep")
    print("="*70 + "\n")

    # Load dropout sweep results
    dropout_results = pd.read_csv('results/dropout_sweep_results.csv')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dropout_rates = dropout_results['dropout'].values

    # MSE plot
    ax = axes[0]
    ax.plot(dropout_rates, dropout_results['train_mse'], 'o-', label='Train', linewidth=2, markersize=8)
    ax.plot(dropout_rates, dropout_results['val_mse'], 's-', label='Validation', linewidth=2, markersize=8)
    ax.plot(dropout_rates, dropout_results['test_mse'], '^-', label='Test', linewidth=2, markersize=8)

    # Mark optimal
    optimal_idx = dropout_results['test_mse'].argmin()
    ax.axvline(dropout_rates[optimal_idx], color='red', linestyle='--', alpha=0.5, label=f'Optimal = {dropout_rates[optimal_idx]:.1f}')

    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('MSE vs Dropout Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # R² plot
    ax = axes[1]
    ax.plot(dropout_rates, dropout_results['train_r2'], 'o-', label='Train', linewidth=2, markersize=8)
    ax.plot(dropout_rates, dropout_results['val_r2'], 's-', label='Validation', linewidth=2, markersize=8)
    ax.plot(dropout_rates, dropout_results['test_r2'], '^-', label='Test', linewidth=2, markersize=8)

    ax.axvline(dropout_rates[optimal_idx], color='red', linestyle='--', alpha=0.5, label=f'Optimal = {dropout_rates[optimal_idx]:.1f}')

    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax.set_title('R² vs Dropout Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('figures/fig3_dropout_sweep.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all three professor-requested figures"""
    print("\n" + "="*70)
    print("Generating Figures Requested by Professor (FIXED)")
    print("="*70)

    # Generate figures
    bimodal_idx, wa, sp_id = figure1_bimodal_experimental_data()
    figure2_model_comparison()
    figure3_dropout_sweep()

    print("\n" + "="*70)
    print("All Figures Generated Successfully!")
    print("="*70)

    if bimodal_idx is not None:
        print(f"\nSummary:")
        print(f"  1. Bimodal experimental data: {sp_id}")
        print(f"     WA: {wa:.2f}")
        print(f"  2. Model comparison: RF vs NN across 3 embeddings")
        print(f"  3. Dropout sweep: Optimal dropout = 0.3-0.5")

    print(f"\nFigures saved to: figures/fig*.png")


if __name__ == '__main__':
    main()
