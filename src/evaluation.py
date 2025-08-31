"""
Model Evaluation Module for Chronic Kidney Disease Prediction

This module provides comprehensive evaluation metrics and visualization tools
for assessing model performance, including:
- Classification metrics
- ROC and PR curves
- Feature importance visualization
- Model comparison plots
- Statistical significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    
    # Advanced metrics
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics['roc_auc'] = None
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", 
                         save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CKD', 'CKD'],
                yticklabels=['No CKD', 'CKD'])
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add metrics text
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve", 
                   save_path=None, figsize=(8, 6)):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_score = roc_auc_score(y_true, y_prob[:, 1])
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve",
                               save_path=None, figsize=(8, 6)):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    
    # Plot curve
    plt.plot(recall, precision, color='darkgreen', lw=2, label='Precision-Recall curve')
    
    # Add baseline (random classifier)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', 
                label=f'Random baseline ({baseline:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_dict, top_n=15, 
                           title="Feature Importance", save_path=None, figsize=(12, 8)):
    """
    Plot feature importance from multiple models.
    
    Args:
        feature_importance_dict: Dictionary of feature importance DataFrames
        top_n: Number of top features to display
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create subplots for each model
    n_models = len(feature_importance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        axes[i].barh(y_pos, top_features['importance'])
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(top_features['feature'])
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
        axes[i].invert_yaxis()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(model_names, metrics_list, metric_name="Accuracy",
                         title="Model Comparison", save_path=None, figsize=(10, 6)):
    """
    Plot comparison of different models based on specified metric.
    
    Args:
        model_names: List of model names
        metrics_list: List of metric values
        metric_name: Name of the metric to compare
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(model_names, metrics_list, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_learning_curves(train_sizes, train_scores, val_scores, 
                         title="Learning Curves", save_path=None, figsize=(10, 6)):
    """
    Plot learning curves showing training and validation scores.
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='r')
    
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='g')
    
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    plt.show()


def generate_evaluation_report(y_true, y_pred, y_prob=None, model_name="Model",
                             save_path=None):
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        model_name: Name of the model
        save_path: Path to save the report
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print metrics
    print(f"\nClassification Metrics:")
    print(f"{'Accuracy':<20}: {metrics['accuracy']:.4f}")
    print(f"{'Precision':<20}: {metrics['precision']:.4f}")
    print(f"{'Recall':<20}: {metrics['recall']:.4f}")
    print(f"{'F1-Score':<20}: {metrics['f1_score']:.4f}")
    print(f"{'Balanced Accuracy':<20}: {metrics['balanced_accuracy']:.4f}")
    print(f"{'Cohen Kappa':<20}: {metrics['cohen_kappa']:.4f}")
    print(f"{'Matthews Corr Coef':<20}: {metrics['matthews_corrcoef']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"{'ROC AUC':<20}: {metrics['roc_auc']:.4f}")
    
    # Print classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=['No CKD', 'CKD']))
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    print("Predicted:")
    print("          No CKD  CKD")
    print(f"Actual No CKD  {cm[0,0]:>6} {cm[0,1]:>4}")
    print(f"      CKD      {cm[1,0]:>6} {cm[1,1]:>4}")
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"EVALUATION REPORT: {model_name}\n")
            f.write("="*60 + "\n\n")
            
            f.write("Classification Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write(f"\nDetailed Classification Report:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(y_true, y_pred, target_names=['No CKD', 'CKD']))
            
            f.write(f"\nConfusion Matrix:\n")
            f.write("-" * 40 + "\n")
            f.write("Predicted:\n")
            f.write("          No CKD  CKD\n")
            f.write(f"Actual No CKD  {cm[0,0]:>6} {cm[0,1]:>4}\n")
            f.write(f"      CKD      {cm[1,0]:>6} {cm[1,1]:>4}\n")
        
        print(f"\nEvaluation report saved to: {save_path}")
    
    return metrics


def evaluate_model(model, X_test, y_test, model_name="Ensemble Model",
                  save_plots=True, output_dir="results/plots"):
    """
    Complete model evaluation with plots and reports.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        save_plots: Whether to save plots
        output_dir: Directory to save outputs
        
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Generate evaluation report
    report_path = f"{output_dir}/evaluation_report_{model_name.lower().replace(' ', '_')}.txt"
    metrics = generate_evaluation_report(y_test, y_pred, y_prob, model_name, report_path)
    
    # Create plots
    if save_plots:
        # Confusion matrix
        cm_path = f"{output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {model_name}", cm_path)
        
        # ROC curve
        roc_path = f"{output_dir}/roc_curve_{model_name.lower().replace(' ', '_')}.png"
        plot_roc_curve(y_test, y_prob, f"ROC Curve - {model_name}", roc_path)
        
        # Precision-Recall curve
        pr_path = f"{output_dir}/pr_curve_{model_name.lower().replace(' ', '_')}.png"
        plot_precision_recall_curve(y_test, y_prob, f"Precision-Recall Curve - {model_name}", pr_path)
    
    return metrics


def plot_results(y_true, y_pred, y_prob, model_name="Model", 
                save_path=None, figsize=(15, 10)):
    """
    Create comprehensive results visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Model Results: {model_name}', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_score = roc_auc_score(y_true, y_prob[:, 1])
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC (AUC = {auc_score:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    axes[1,0].plot(recall, precision, color='darkgreen', lw=2)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    axes[1,0].grid(True, alpha=0.3)
    
    # Metrics Summary
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics_text = f"""Accuracy: {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1-Score: {metrics['f1_score']:.3f}
ROC AUC: {metrics.get('roc_auc', 'N/A')}"""
    
    axes[1,1].text(0.1, 0.5, metrics_text, transform=axes[1,1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1,1].set_title('Performance Metrics')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully!")
    print("Use the functions to evaluate your CKD prediction models.")
