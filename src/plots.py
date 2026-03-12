
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target_distribution(y):
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=40)
    plt.xlabel("Critical Temperature (K)")
    plt.ylabel("Count")
    plt.title("Distribution of Superconducting Critical Temperature")
    plt.tight_layout()
    plt.show()


def plot_pca_variance(pca):
    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained) + 1), explained, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Cumulative Explained Variance")
    plt.tight_layout()
    plt.show()


def plot_clusters(X_pca, clusters, y, high_tc_quantile: float = 0.9):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.6)
    plt.colorbar(scatter, label="Cluster")

    high_tc_mask = y >= y.quantile(high_tc_quantile)
    plt.scatter(
        X_pca[high_tc_mask, 0],
        X_pca[high_tc_mask, 1],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.2,
        label="Top 10% Tc",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means Clusters in PCA Space")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pc_vs_target(X_pca, y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(X_pca[:, 0], y, alpha=0.5)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("Critical Temperature (K)")
    ax1.set_title("PC1 vs Critical Temperature")

    ax2.scatter(X_pca[:, 1], y, alpha=0.5)
    ax2.set_xlabel("PC2")
    ax2.set_ylabel("Critical Temperature (K)")
    ax2.set_title("PC2 vs Critical Temperature")
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred):
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.set_xlabel("Actual Tc (K)")
    ax1.set_ylabel("Predicted Tc (K)")
    ax1.set_title("Predicted vs Actual Critical Temperature")

    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(0)
    ax2.set_xlabel("Predicted Tc (K)")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.set_title("Residuals vs Predicted Tc")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_table: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    ordered = feature_table.sort_values("importance")
    plt.barh(ordered["feature"], ordered["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances from Random Forest")
    plt.tight_layout()
    plt.show()
