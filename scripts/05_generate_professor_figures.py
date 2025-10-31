#!/usr/bin/env python3
"""
Generate Specific Figures Requested by Professor
==================================================

Creates three key figures:
1. Bimodal distribution example (WA prediction between two peaks)
2. Generalized model comparison (RF vs NN across embeddings)
3. Dropout hyperparameter sweep

Usage:
    python scripts/05_generate_professor_figures.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

from data_utils import load_plm_embeddings
from models import ProteinClassifierNN


def figure1_bimodal_distribution_example():
    """
    Find and plot a signal peptide where the model predicts bimodal distribution.

    The WA prediction falls between two peaks, showing uncertainty.
    """
    print("\n" + "="*70)
    print("Figure 1: Bimodal Distribution Example")
    print("="*70 + "\n")

    # Load best model data
    X_train, X_test, y_train, y_test = load_plm_embeddings('ginkgo-AA0-650M')

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Training neural network...")
    classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')
    classifier.create_bins(y_train)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    classifier.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        architecture='single_layer',
        units=[256, 128],
        dropout_rates=[0.3],
        l2_reg=0.001,
        learning_rate=0.0005,
        batch_size=32,
        epochs=100,
        verbose=0
    )

    # Get probability distributions for test set
    probs = classifier.model.predict(X_test_scaled, verbose=0)

    # Find examples with bimodal distributions
    # Look for cases where two non-adjacent bins both have >0.2 probability
    bimodal_indices = []
    for i, prob_dist in enumerate(probs):
        sorted_probs = np.sort(prob_dist)[::-1]
        # Check if top 2 probabilities are both >0.2 and have a gap
        if sorted_probs[0] > 0.25 and sorted_probs[1] > 0.20:
            # Find indices of top 2 bins
            top_bins = np.argsort(prob_dist)[::-1][:2]
            if abs(top_bins[0] - top_bins[1]) >= 2:  # Non-adjacent bins
                bimodal_indices.append(i)

    if len(bimodal_indices) == 0:
        print("No bimodal examples found, showing high-uncertainty example instead")
        # Find example with highest entropy (most uncertain)
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        bimodal_idx = np.argmax(entropies)
    else:
        bimodal_idx = bimodal_indices[0]

    # Get data for this example
    prob_dist = probs[bimodal_idx]
    true_wa = y_test[bimodal_idx]
    pred_wa = np.sum(prob_dist * classifier.bin_centers)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Probability distribution
    bin_edges = classifier.bin_edges
    bin_centers = classifier.bin_centers

    ax1.bar(bin_centers, prob_dist, width=0.8, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=1.5)
    ax1.axvline(true_wa, color='red', linestyle='--', linewidth=2.5,
                label=f'True WA = {true_wa:.2f}', alpha=0.8)
    ax1.axvline(pred_wa, color='green', linestyle='--', linewidth=2.5,
                label=f'Predicted WA = {pred_wa:.2f}', alpha=0.8)

    ax1.set_xlabel('WA Score (Secretion Efficiency)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Bimodal Prediction Distribution\nWA Prediction Between Two Peaks',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(prob_dist) * 1.15)

    # Add annotation showing the issue
    top_bins = np.argsort(prob_dist)[::-1][:2]
    ax1.annotate('', xy=(bin_centers[top_bins[0]], prob_dist[top_bins[0]]),
                xytext=(bin_centers[top_bins[1]], prob_dist[top_bins[1]]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(np.mean([bin_centers[top_bins[0]], bin_centers[top_bins[1]]]),
             max(prob_dist[top_bins]) * 0.5,
             'Model\nUncertain', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=10, fontweight='bold')

    # Right: Show why this happens - comparing unimodal vs bimodal
    # Generate example distributions
    x = np.linspace(0, 10, 100)

    # Unimodal (confident)
    unimodal = np.exp(-((x - 5) ** 2) / (2 * 0.5 ** 2))
    unimodal /= unimodal.sum()

    # Bimodal (uncertain)
    bimodal_ex = (np.exp(-((x - 3) ** 2) / (2 * 0.8 ** 2)) +
                  np.exp(-((x - 7) ** 2) / (2 * 0.8 ** 2)))
    bimodal_ex /= bimodal_ex.sum() * 5  # Normalize for visibility

    ax2.fill_between(x, unimodal, alpha=0.5, color='green',
                      label='Unimodal (Confident)', linewidth=2)
    ax2.plot(x, unimodal, color='darkgreen', linewidth=2.5)

    ax2.fill_between(x, bimodal_ex, alpha=0.5, color='orange',
                      label='Bimodal (Uncertain)', linewidth=2)
    ax2.plot(x, bimodal_ex, color='darkorange', linewidth=2.5)

    # Mark predicted values
    unimodal_pred = 5.0
    bimodal_pred = (3 + 7) / 2  # Mean of two peaks

    ax2.axvline(unimodal_pred, color='darkgreen', linestyle=':', linewidth=2,
                alpha=0.7)
    ax2.axvline(bimodal_pred, color='darkorange', linestyle=':', linewidth=2,
                alpha=0.7)

    ax2.set_xlabel('WA Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('Unimodal vs Bimodal Predictions\nWA Not Always Most Likely Bin',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text annotation
    ax2.text(0.5, 0.95, 'When model is uncertain, predicted WA\nfalls between peaks (not at peak)',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent.parent / 'figures' / 'prof_fig1_bimodal_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    return bimodal_idx, true_wa, pred_wa


def figure2_model_comparison():
    """
    Generalized Figure 1a: Compare RF vs NN across different embeddings.

    Line style = model type (solid=NN, dashed=RF)
    Color = embedding type (blue=ESM-650M, orange=ESM-3B, green=Ginkgo)
    """
    print("\n" + "="*70)
    print("Figure 2: Generalized Model Comparison")
    print("="*70 + "\n")

    # Load results from CSVs
    rf_results = pd.read_csv(Path(__file__).parent.parent / 'results' / 'rf_baseline_results.csv')
    nn_results = pd.read_csv(Path(__file__).parent.parent / 'results' / 'nn_classifier_results.csv')

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['test_mse', 'test_r2', 'test_spearman']
    titles = ['Mean Squared Error (MSE)', 'R² Score', 'Spearman Correlation (ρ)']
    ylabels = ['MSE (lower is better)', 'R²', 'Spearman ρ']

    # Colors for embeddings
    colors = {
        'esm2-650M': '#1f77b4',  # Blue
        'esm2-3B': '#ff7f0e',    # Orange
        'ginkgo-AA0-650M': '#2ca02c'  # Green
    }

    embedding_labels = {
        'esm2-650M': 'ESM-2 650M',
        'esm2-3B': 'ESM-2 3B',
        'ginkgo-AA0-650M': 'Ginkgo AA0'
    }

    x_pos = {'esm2-650M': 0, 'esm2-3B': 1, 'ginkgo-AA0-650M': 2}

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]

        # Plot each embedding
        for emb in ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']:
            x = x_pos[emb]

            # Get values
            rf_val = rf_results[rf_results['embedding_model'] == emb][metric].values[0]
            nn_val = nn_results[nn_results['embedding_model'] == emb][metric].values[0]

            color = colors[emb]

            # Plot RF (dashed line, circles)
            if idx == 0 or emb == 'esm2-650M':  # Only label once
                rf_label = 'Random Forest' if emb == 'esm2-650M' else None
                nn_label = 'Neural Network' if emb == 'esm2-650M' else None
            else:
                rf_label = nn_label = None

            # Plot points
            ax.plot(x, rf_val, 'o', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.7)
            ax.plot(x, nn_val, 's', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.7)

        # Connect points for each model
        for emb_idx, emb in enumerate(['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']):
            if emb_idx < 2:
                next_emb = ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M'][emb_idx + 1]

                rf_vals = [
                    rf_results[rf_results['embedding_model'] == emb][metric].values[0],
                    rf_results[rf_results['embedding_model'] == next_emb][metric].values[0]
                ]
                nn_vals = [
                    nn_results[nn_results['embedding_model'] == emb][metric].values[0],
                    nn_results[nn_results['embedding_model'] == next_emb][metric].values[0]
                ]

                # RF line (dashed)
                ax.plot([x_pos[emb], x_pos[next_emb]], rf_vals, '--',
                       color='gray', linewidth=2, alpha=0.5,
                       label='Random Forest' if emb_idx == 0 else None)

                # NN line (solid)
                ax.plot([x_pos[emb], x_pos[next_emb]], nn_vals, '-',
                       color='gray', linewidth=2.5, alpha=0.7,
                       label='Neural Network' if emb_idx == 0 else None)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([embedding_labels[e] for e in ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']],
                           rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

    # Add overall title
    fig.suptitle('Model Comparison Across Protein Language Model Embeddings\n' +
                 'Line Style: Solid=Neural Network, Dashed=Random Forest | ' +
                 'Color: Embedding Type',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent.parent / 'figures' / 'prof_fig2_model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def figure3_dropout_sweep():
    """
    Systematic dropout hyperparameter sweep.

    Tests dropout rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    """
    print("\n" + "="*70)
    print("Figure 3: Dropout Hyperparameter Sweep")
    print("="*70 + "\n")

    # Load data
    X_train, X_test, y_train, y_test = load_plm_embeddings('ginkgo-AA0-650M')

    # Scale features
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train_scaled, X_val_scaled, y_train_split, y_val_split = train_test_split(
        X_train_full_scaled, y_train, test_size=0.2, random_state=42
    )

    # Test different dropout rates
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for dropout in dropout_rates:
        print(f"  Testing dropout = {dropout}...")

        classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')
        classifier.create_bins(y_train)

        classifier.fit(
            X_train_scaled, y_train_split,
            X_val=X_val_scaled, y_val=y_val_split,
            architecture='single_layer',
            units=[256, 128],
            dropout_rates=[dropout],
            l2_reg=0.001,
            learning_rate=0.0005,
            batch_size=32,
            epochs=100,
            verbose=0
        )

        # Evaluate
        from eval_utils import compute_regression_metrics

        y_train_pred = classifier.predict(X_train_scaled)
        y_val_pred = classifier.predict(X_val_scaled)
        y_test_pred = classifier.predict(X_test_scaled)

        train_metrics = compute_regression_metrics(y_train_split, y_train_pred)
        val_metrics = compute_regression_metrics(y_val_split, y_val_pred)
        test_metrics = compute_regression_metrics(y_test, y_test_pred)

        results.append({
            'dropout': dropout,
            'train_mse': train_metrics['mse'],
            'val_mse': val_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'test_r2': test_metrics['r2']
        })

        print(f"    Train MSE: {train_metrics['mse']:.4f}, Val MSE: {val_metrics['mse']:.4f}, " +
              f"Test MSE: {test_metrics['mse']:.4f}")

    results_df = pd.DataFrame(results)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MSE vs dropout
    ax1.plot(results_df['dropout'], results_df['train_mse'], 'o-',
            label='Training', linewidth=2.5, markersize=10, color='#1f77b4')
    ax1.plot(results_df['dropout'], results_df['val_mse'], 's-',
            label='Validation', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax1.plot(results_df['dropout'], results_df['test_mse'], '^-',
            label='Test', linewidth=2.5, markersize=10, color='#2ca02c')

    # Mark optimal
    best_idx = results_df['val_mse'].idxmin()
    best_dropout = results_df.loc[best_idx, 'dropout']
    best_val_mse = results_df.loc[best_idx, 'val_mse']

    ax1.axvline(best_dropout, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Optimal: {best_dropout}')
    ax1.plot(best_dropout, best_val_mse, '*', color='red',
            markersize=20, markeredgecolor='black', markeredgewidth=1.5)

    ax1.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('MSE vs Dropout Rate\nLower is Better',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 0.55)

    # Right: R² vs dropout
    ax2.plot(results_df['dropout'], results_df['train_r2'], 'o-',
            label='Training', linewidth=2.5, markersize=10, color='#1f77b4')
    ax2.plot(results_df['dropout'], results_df['val_r2'], 's-',
            label='Validation', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax2.plot(results_df['dropout'], results_df['test_r2'], '^-',
            label='Test', linewidth=2.5, markersize=10, color='#2ca02c')

    # Mark optimal
    best_val_r2 = results_df.loc[best_idx, 'val_r2']

    ax2.axvline(best_dropout, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Optimal: {best_dropout}')
    ax2.plot(best_dropout, best_val_r2, '*', color='red',
            markersize=20, markeredgecolor='black', markeredgewidth=1.5)

    ax2.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² vs Dropout Rate\nHigher is Better',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 0.55)

    # Add findings box
    fig.text(0.5, 0.02,
            f'Optimal dropout rate: {best_dropout} | ' +
            f'Test MSE: {results_df.loc[best_idx, "test_mse"]:.4f} | ' +
            f'Test R²: {results_df.loc[best_idx, "test_r2"]:.4f}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    save_path = Path(__file__).parent.parent / 'figures' / 'prof_fig3_dropout_sweep.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'dropout_sweep_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved results: {results_path}")

    plt.close()

    return results_df


def main():
    """Generate all requested figures."""
    print("\n" + "="*70)
    print("Generating Figures Requested by Professor")
    print("="*70)

    # Figure 1: Bimodal distribution
    bimodal_idx, true_wa, pred_wa = figure1_bimodal_distribution_example()

    # Figure 2: Model comparison
    figure2_model_comparison()

    # Figure 3: Dropout sweep
    dropout_results = figure3_dropout_sweep()

    print("\n" + "="*70)
    print("All Figures Generated Successfully!")
    print("="*70)
    print("\nSummary:")
    print(f"  1. Bimodal distribution example: Test sample {bimodal_idx}")
    print(f"     True WA: {true_wa:.2f}, Predicted WA: {pred_wa:.2f}")
    print(f"  2. Model comparison: RF vs NN across 3 embeddings")
    print(f"  3. Dropout sweep: Optimal dropout = {dropout_results.loc[dropout_results['val_mse'].idxmin(), 'dropout']}")
    print(f"\nFigures saved to: figures/prof_fig*.png")
    print()


if __name__ == '__main__':
    main()
