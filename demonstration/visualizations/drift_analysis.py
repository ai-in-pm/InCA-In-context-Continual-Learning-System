"""
Visualization script for concept drift analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

def plot_drift_scores(drift_scores: pd.Series, threshold: float = 0.3, title: str = "Drift Scores Over Time"):
    """Plot drift scores with threshold line."""
    plt.figure(figsize=(12, 6))
    plt.plot(drift_scores.index, drift_scores.values, label='Drift Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_embedding_distribution(embeddings: np.ndarray, labels: list, title: str = "Embedding Distribution"):
    """Plot t-SNE visualization of embeddings."""
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', alpha=0.6)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

def plot_distribution_comparison(reference: np.ndarray, current: np.ndarray, 
                               title: str = "Distribution Comparison"):
    """Plot distribution comparison between reference and current data."""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(reference, bins=30, alpha=0.5, label='Reference', density=True)
    plt.hist(current, bins=30, alpha=0.5, label='Current', density=True)
    
    # Plot KDE
    sns.kdeplot(data=reference, label='Reference KDE')
    sns.kdeplot(data=current, label='Current KDE')
    
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_drift_metrics_over_time(metrics: pd.DataFrame, title: str = "Drift Metrics Over Time"):
    """Plot multiple drift metrics over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Drift Score and Severity
    ax1.plot(metrics.index, metrics['drift_score'], label='Drift Score')
    ax1.plot(metrics.index, metrics['severity'], label='Severity')
    ax1.set_title("Drift Score and Severity Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Distribution Distance
    ax2.plot(metrics.index, metrics['distribution_distance'], label='Distribution Distance')
    ax2.set_title("Distribution Distance Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Distance")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Example drift scores
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    drift_scores = pd.Series(
        np.random.normal(0.2, 0.1, 30).clip(0, 1),
        index=dates
    )
    plot_drift_scores(drift_scores)
    
    # Example embeddings
    n_samples = 300
    embeddings = np.random.randn(n_samples, 50)  # 50-dimensional embeddings
    labels = ['reference'] * (n_samples // 2) + ['current'] * (n_samples // 2)
    plot_embedding_distribution(embeddings, labels)
    
    # Example distributions
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0.5, 1.2, 1000)
    plot_distribution_comparison(reference, current)
    
    # Example drift metrics
    metrics = pd.DataFrame({
        'drift_score': np.random.uniform(0, 0.5, 30),
        'severity': np.random.uniform(0, 0.7, 30),
        'distribution_distance': np.random.uniform(0.1, 0.4, 30)
    }, index=dates)
    plot_drift_metrics_over_time(metrics)
