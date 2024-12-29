import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DistributionVisualizer:
    """Tools for visualizing class distributions and classification results."""
    
    @staticmethod
    def plot_mahalanobis_heatmap(distances: Dict[str, Dict[str, float]], 
                                title: str = "Mahalanobis Distances"):
        """Plot heatmap of Mahalanobis distances between texts and classes."""
        # Convert distances to matrix form
        classes = list(distances.keys())
        texts = list(distances[classes[0]].keys())
        
        matrix = np.zeros((len(texts), len(classes)))
        for i, text in enumerate(texts):
            for j, class_name in enumerate(classes):
                matrix[i, j] = distances[class_name][text]
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, 
                   xticklabels=classes,
                   yticklabels=[f"Text {i+1}" for i in range(len(texts))],
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd')
        plt.title(title)
        plt.xlabel("Classes")
        plt.ylabel("Texts")
        return plt.gcf()
    
    @staticmethod
    def plot_embedding_clusters(embeddings: Dict[str, List[List[float]]],
                              method: str = 'pca',
                              perplexity: int = 30):
        """
        Plot embeddings clusters using dimensionality reduction.
        
        Args:
            embeddings: Dictionary mapping class names to lists of embedding vectors
            method: 'pca' or 'tsne'
            perplexity: Perplexity parameter for t-SNE
        """
        # Combine all embeddings and create labels
        all_embeddings = []
        labels = []
        for class_name, class_embeddings in embeddings.items():
            all_embeddings.extend(class_embeddings)
            labels.extend([class_name] * len(class_embeddings))
        
        X = np.array(all_embeddings)
        
        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # t-SNE
            reducer = TSNE(n_components=2, perplexity=perplexity)
        
        X_reduced = reducer.fit_transform(X)
        
        # Plot
        plt.figure(figsize=(10, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            plt.scatter(X_reduced[mask, 0], 
                       X_reduced[mask, 1],
                       c=[color],
                       label=label,
                       alpha=0.6)
        
        plt.title(f"Embedding Clusters ({method.upper()})")
        plt.legend()
        return plt.gcf()
    
    @staticmethod
    def plot_confidence_distribution(confidences: Dict[str, List[float]],
                                   bins: int = 30):
        """Plot histogram of confidence scores for each class."""
        plt.figure(figsize=(12, 6))
        
        n_classes = len(confidences)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        for (class_name, scores), color in zip(confidences.items(), colors):
            plt.hist(scores,
                    bins=bins,
                    alpha=0.3,
                    color=color,
                    label=class_name)
        
        plt.title("Distribution of Confidence Scores")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.legend()
        return plt.gcf()
    
    @staticmethod
    def plot_performance_metrics(metrics: Dict[str, List[Tuple[float, float]]]):
        """Plot performance metrics over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for operation, values in metrics.items():
            times, durations = zip(*values)
            ax1.plot(times, durations, label=operation)
        
        ax1.set_title("Operation Duration Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Duration (ms)")
        ax1.legend()
        
        # Add memory usage if available
        if 'memory_mb' in metrics:
            times, memory = zip(*metrics['memory_mb'])
            ax2.plot(times, memory, label='Memory Usage', color='red')
            ax2.set_title("Memory Usage Over Time")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Memory (MB)")
        
        plt.tight_layout()
        return plt.gcf()
