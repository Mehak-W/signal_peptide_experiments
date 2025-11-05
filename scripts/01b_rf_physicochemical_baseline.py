#!/usr/bin/env python3
"""
Random Forest Baseline on Physicochemical Features
===================================================

Reproduces Grasso et al. (2023) original approach using physicochemical features.
This validates the improvement claims by comparing:
- RF on physicochemical features (Grasso's original)
- RF on PLM embeddings (our baseline)
- NN on PLM embeddings (our improvement)

Usage:
    python scripts/01b_rf_physicochemical_baseline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from eval_utils import compute_regression_metrics
from plot_utils import plot_predictions_vs_actual

# Set random seed
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_physicochemical_features():
    """
    Load physicochemical features from Grasso dataset.
    Uses same train/test split as PLM embeddings.

    Returns:
    --------
    X_train, X_test : np.ndarray
        Feature matrices
    y_train, y_test : np.ndarray
        Target values (WA scores)
    feature_names : list
        Names of physicochemical features
    """
    # Load from xlsx file
    xlsx_path = Path(__file__).parent.parent / 'data' / 'sb2c00328_si_011.xlsx'

    # Load WT sequences sheet (has physicochemical features)
    df_features = pd.read_excel(xlsx_path, sheet_name='WT sequences')

    # Get train/test split from parquet files (same split as PLM embeddings)
    train_parquet = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'trainAA_esm2-650M.parquet')
    test_parquet = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'testAA_esm2-650M.parquet')

    # Get physicochemical features from WT sequences sheet
    # Exclude non-feature columns
    exclude_cols = ['Index', 'UniprotID', 'Locus', 'gene',
                    'N-start', 'N-end', 'H-start', 'H-end', 'C-start', 'C-end',
                    'SP_nt', 'N_nt', 'H_nt', 'C_nt', 'Ac_nt',
                    'SP_aa', 'N_aa', 'H_aa', 'C_aa', 'Ac_aa']

    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    print(f"\nPhysicochemical features: {len(feature_cols)}")

    # Create mapping from sequence to features
    seq_to_features = {}
    for idx, row in df_features.iterrows():
        seq = row['SP_aa']
        features = row[feature_cols].values
        seq_to_features[seq] = features

    # Extract features for train/test based on sequences in parquet files
    X_train_list = []
    y_train_list = []
    for idx, row in train_parquet.iterrows():
        seq = row['sequence']
        if seq in seq_to_features:
            X_train_list.append(seq_to_features[seq])
            y_train_list.append(row['WA'])

    X_test_list = []
    y_test_list = []
    for idx, row in test_parquet.iterrows():
        seq = row['sequence']
        if seq in seq_to_features:
            X_test_list.append(seq_to_features[seq])
            y_test_list.append(row['WA'])

    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    print(f"Total WT sequences: {len(df_features)}")
    print(f"Matched - Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"\nFeature matrix shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_rf_physicochemical():
    """Train Random Forest on physicochemical features"""
    print("\n" + "="*70)
    print("Random Forest Baseline on Physicochemical Features")
    print("="*70 + "\n")

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_physicochemical_features()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with Grasso's hyperparameters
    print("Training Random Forest...")
    print("  Hyperparameters: n_estimators=100, max_depth=20, min_samples_split=5")

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nEvaluating...")
    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    # Print results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"\nTrain Metrics:")
    print(f"  MSE:      {train_metrics['mse']:.4f}")
    print(f"  RMSE:     {train_metrics['rmse']:.4f}")
    print(f"  MAE:      {train_metrics['mae']:.4f}")
    print(f"  R²:       {train_metrics['r2']:.4f}")
    print(f"  Spearman: {train_metrics['spearman']:.4f}")

    print(f"\nTest Metrics:")
    print(f"  MSE:      {test_metrics['mse']:.4f}")
    print(f"  RMSE:     {test_metrics['rmse']:.4f}")
    print(f"  MAE:      {test_metrics['mae']:.4f}")
    print(f"  R²:       {test_metrics['r2']:.4f}")
    print(f"  Spearman: {test_metrics['spearman']:.4f}")

    # Save results
    results = {
        'model': ['Random Forest'],
        'features': ['physicochemical'],
        'train_mse': [train_metrics['mse']],
        'train_rmse': [train_metrics['rmse']],
        'train_mae': [train_metrics['mae']],
        'train_r2': [train_metrics['r2']],
        'train_spearman': [train_metrics['spearman']],
        'train_spearman_pval': [train_metrics['spearman_pval']],
        'train_pearson': [train_metrics['pearson']],
        'train_pearson_pval': [train_metrics['pearson_pval']],
        'test_mse': [test_metrics['mse']],
        'test_rmse': [test_metrics['rmse']],
        'test_mae': [test_metrics['mae']],
        'test_r2': [test_metrics['r2']],
        'test_spearman': [test_metrics['spearman']],
        'test_spearman_pval': [test_metrics['spearman_pval']],
        'test_pearson': [test_metrics['pearson']],
        'test_pearson_pval': [test_metrics['pearson_pval']]
    }

    results_df = pd.DataFrame(results)

    # Save to CSV
    output_csv = Path('results/rf_physicochemical_baseline.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved results: {output_csv}")

    # Update main RF results file to include this
    try:
        main_results = pd.read_csv('results/rf_baseline_results.csv')
        # Add physicochemical baseline if not already there
        if 'physicochemical' not in main_results['embedding_model'].values:
            new_row = {
                'model': 'Random Forest',
                'embedding_model': 'physicochemical',
                **{k: v[0] for k, v in results.items() if k not in ['model', 'features']}
            }
            main_results = pd.concat([main_results, pd.DataFrame([new_row])], ignore_index=True)
            main_results.to_csv('results/rf_baseline_results.csv', index=False)
            print(f"✓ Updated: results/rf_baseline_results.csv")
    except Exception as e:
        print(f"Note: Could not update main results file: {e}")

    # Plot predictions
    print("\nGenerating prediction plot...")
    fig = plot_predictions_vs_actual(
        y_test, y_test_pred,
        title='RF on Physicochemical Features (Grasso Baseline)'
    )

    output_fig = Path('figures/rf_physicochemical_predictions.png')
    fig.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: {output_fig}")
    plt.close()

    # Feature importance (top 20)
    print("\nTop 20 most important features:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

    return results_df


def main():
    """Main execution"""
    results = train_rf_physicochemical()

    print("\n" + "="*70)
    print("Comparison to Grasso et al. (2023)")
    print("="*70)
    print(f"\nGrasso reported: MSE ≈ 1.22")
    print(f"Our reproduction: MSE = {results['test_mse'].values[0]:.4f}")

    if results['test_mse'].values[0] <= 1.3:
        print("✓ Successfully reproduced Grasso baseline!")
    else:
        print("⚠ Higher MSE than reported - may need hyperparameter tuning")


if __name__ == '__main__':
    main()
