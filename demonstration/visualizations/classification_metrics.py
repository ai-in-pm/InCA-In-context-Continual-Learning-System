"""
Visualization script for classification metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confidence_distribution(confidences: pd.DataFrame, title: str = "Confidence Distribution"):
    """Plot confidence score distribution for each class."""
    plt.figure(figsize=(12, 6))
    for class_name in confidences.columns:
        sns.kdeplot(data=confidences[class_name], label=class_name)
    plt.title(title)
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, classes: list, title: str = "Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_performance_metrics(metrics: pd.DataFrame, title: str = "Performance Over Time"):
    """Plot performance metrics over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Accuracy and F1 Score
    ax1.plot(metrics.index, metrics['accuracy'], label='Accuracy')
    ax1.plot(metrics.index, metrics['f1_score'], label='F1 Score')
    ax1.set_title("Accuracy and F1 Score Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Response Time
    ax2.plot(metrics.index, metrics['response_time'], label='Response Time')
    ax2.set_title("Response Time Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Response Time (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(class_counts: dict, title: str = "Class Distribution"):
    """Plot class distribution bar chart."""
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Example confidence scores
    confidences = pd.DataFrame({
        'technical': np.random.beta(5, 2, 100),
        'business': np.random.beta(2, 5, 100),
        'general': np.random.beta(3, 3, 100)
    })
    plot_confidence_distribution(confidences)
    
    # Example confusion matrix
    classes = ['technical', 'business', 'general']
    cm = np.array([
        [85, 10, 5],
        [8, 90, 2],
        [4, 3, 93]
    ])
    plot_confusion_matrix(cm, classes)
    
    # Example performance metrics
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    metrics = pd.DataFrame({
        'accuracy': np.random.uniform(0.8, 0.95, 30),
        'f1_score': np.random.uniform(0.75, 0.9, 30),
        'response_time': np.random.uniform(100, 300, 30)
    }, index=dates)
    plot_performance_metrics(metrics)
    
    # Example class distribution
    class_counts = {
        'technical': 1200,
        'business': 800,
        'general': 1000,
        'other': 500
    }
    plot_class_distribution(class_counts)
