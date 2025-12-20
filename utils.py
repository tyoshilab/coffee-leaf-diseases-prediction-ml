"""
Utility functions for Coffee Leaf Disease Prediction notebooks.

This module contains reusable functions for:
- Data loading and feature extraction
- Model evaluation and visualization
- Custom estimators for multi-label classification
"""

import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import label_binarize


# =============================================================================
# Data Loading and Feature Extraction
# =============================================================================

def convert_to_single_label(row):
    """
    Convert multi-label row to single label.
    
    Parameters
    ----------
    row : pd.Series
        Row containing 'miner', 'phoma', 'rust' columns.
    
    Returns
    -------
    str
        Single label string.
    """
    if row['miner'] == 1:
        return 'miner'
    elif row['phoma'] == 1:
        return 'phoma'
    elif row['rust'] == 1:
        return 'rust'
    else:
        return 'nodisease'


def rgb_to_cmy(rgb_image):
    # CMY = 1 - RGB
    cmy_image = 1.0 - rgb_image
    return cmy_image

def extract_color_features(image):
    features = []
    
    # RGB features (6)
    for channel in range(3):  # R, G, B
        channel_data = image[:, :, channel]
        features.append(np.mean(channel_data))  # Mean
        features.append(np.std(channel_data))   # Standard deviation
    
    # CMY features (6)
    cmy_image = rgb_to_cmy(image)
    for channel in range(3):  # C, M, Y
        channel_data = cmy_image[:, :, channel]
        features.append(np.mean(channel_data))  # Mean
        features.append(np.std(channel_data))   # Standard deviation
    
    return np.array(features)

def load_and_extract_features(train_or_test, resize_shape, use_raw_data=False, single_label=False):
    """
    Load images and extract features.
    
    Parameters
    ----------
    train_or_test : str
        'train' or 'test'.
    resize_shape : tuple
        Target size for image resizing (width, height).
    use_raw_data : bool, optional
        If True, use raw pixel data instead of color features.
    single_label : bool, optional
        If True, convert multi-label to single label.
    
    Returns
    -------
    features_array : np.ndarray
        Extracted features.
    labels : pd.DataFrame or pd.Series
        Labels for each sample.
    """
    labels_df = pd.read_csv(f'dataset/{train_or_test}_classes.csv')
    image_dir = f'dataset/coffee-leaf-diseases/{train_or_test}/images'

    features_list = []
    valid_indices = []
    
    for idx, row in labels_df.iterrows():
        img_path = os.path.join(image_dir, f"{row['id']}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_resized = img.resize(resize_shape, Image.Resampling.BILINEAR)
            img_array = np.array(img_resized).astype('float32') / 255.0
            if use_raw_data:
                features = img_array
            else:
                features = extract_color_features(img_array)
            features_list.append(features)
            valid_indices.append(idx)
        else:
            print(f"Warning: {img_path} not found")
    
    features_array = np.array(features_list)
    if use_raw_data:
        features_array = features_array.reshape(features_array.shape[0], -1)
    labels = labels_df.loc[valid_indices].reset_index(drop=True)
    labels = labels.drop(columns=['id'], axis=1)
    
    if single_label:
        labels = labels_df.loc[valid_indices].reset_index(drop=True)
        labels['label'] = labels.apply(convert_to_single_label, axis=1)
        labels = labels['label']
    
    return features_array, labels

# =============================================================================
# Model Evaluation
# =============================================================================

def show_evaluation_results(model_type, pred, actual):
    """
    Display evaluation metrics for multi-label classification.
    
    Parameters
    ----------
    model_type : str
        Name of the model being evaluated.
    pred : array-like
        Predicted labels.
    actual : array-like
        Actual labels.
    """
    print(f"\n=== {model_type} Overall Metrics ===")
    print("Accuracy (subset accuracy):", accuracy_score(actual, pred))
    print("Precision (micro):", precision_score(actual, pred, average='micro', zero_division=0))
    print("Recall (micro):", recall_score(actual, pred, average='micro', zero_division=0))
    print("F1-score (micro):", f1_score(actual, pred, average='micro', zero_division=0))
    print("Precision (macro):", precision_score(actual, pred, average='macro', zero_division=0))
    print("Recall (macro):", recall_score(actual, pred, average='macro', zero_division=0))
    print("F1-score (macro):", f1_score(actual, pred, average='macro', zero_division=0))


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix_single_label(model_type, pred, actual, labels, target_name, figsize=(6, 5)):
    """
    Plot confusion matrix heatmap for single-label classification.
    
    Parameters
    ----------
    model_type : str
        Name of the model being evaluated.
    pred : array-like
        Predicted labels.
    actual : array-like
        Actual labels.
    labels : list
        List of class labels.
    target_name : str
        Name of the target set (e.g., 'Validation Set', 'Test Set').
    figsize : tuple, optional
        Figure size. Default is (6, 5).
    """
    cm = confusion_matrix(actual, pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix of {target_name} - {model_type}')
    plt.show()


def plot_roc_curve_single_label(model_type, model, pred_target, actual, label_encoder, target_name):
    """
    Plot ROC-AUC curves for single-label multi-class classification.
    
    Parameters
    ----------
    model_type : str
        Name of the model being evaluated.
    model : estimator
        Trained model with predict_proba method.
    pred_target : array-like
        Features to predict on.
    actual : array-like
        Actual labels (not encoded).
    label_encoder : LabelEncoder
        Fitted label encoder.
    target_name : str
        Name of the target set (e.g., 'Validation Set', 'Test Set').
    """
    y_score = model.predict_proba(pred_target)
    y_bin = label_binarize(actual, classes=label_encoder.classes_)
    all_auc = []
    
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{class_name} (AUC = {roc_auc:.4f})'
        )
        
    macro_auc = sum(all_auc) / len(all_auc)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type} ROC Curve ({target_name})\nOverall Macro-AUC: {macro_auc:.4f}')
    plt.legend()
    plt.show()


def plot_confusion_matrix(model_type, pred, actual, labels=None, figsize=(15, 4)):
    """
    Plot confusion matrix heatmaps for each label in multi-label classification.
    
    Parameters
    ----------
    model_type : str
        Name of the model being evaluated.
    pred : array-like
        Predicted labels.
    actual : array-like or pd.DataFrame
        Actual labels.
    labels : list, optional
        List of label names. If None, uses columns from actual if DataFrame.
    figsize : tuple, optional
        Figure size. Default is (15, 4).
    """
    if labels is None:
        if hasattr(actual, 'columns'):
            labels = actual.columns
        else:
            labels = [f'Label {i}' for i in range(actual.shape[1])]
    
    pred_array = np.array(pred)
    actual_array = np.array(actual)
    
    plt.figure(figsize=figsize)
    
    for i, label_name in enumerate(labels):
        y_true_label = actual_array[:, i]
        y_pred_label = pred_array[:, i]
        
        cm = confusion_matrix(y_true_label, y_pred_label)
        
        plt.subplot(1, len(labels), i + 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=['Not ' + label_name, label_name],
            yticklabels=['Not ' + label_name, label_name]
        )
        plt.title(f'{model_type} Confusion Matrix - {label_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model_type, model, pred_target, actual, target_name, labels=None):
    """
    Plot ROC-AUC curves for multi-label classification.
    
    Parameters
    ----------
    model_type : str
        Name of the model being evaluated.
    model : estimator
        Trained model with predict_proba method.
    pred_target : array-like
        Features to predict on.
    actual : array-like or pd.DataFrame
        Actual labels.
    target_name : str
        Name of the target set (e.g., 'Validation Set', 'Test Set').
    labels : list, optional
        List of label names. If None, uses columns from actual if DataFrame.
    """
    if labels is None:
        if hasattr(actual, 'columns'):
            labels = actual.columns
        else:
            labels = [f'Label {i}' for i in range(actual.shape[1])]
    
    actual_array = np.array(actual)
    y_score = model.predict_proba(pred_target)
    
    all_auc = []
    
    is_multioutput_style = False
    if isinstance(model, SKPipeline) and isinstance(model.steps[-1][1], MultiOutputClassifier):
        is_multioutput_style = True
    elif isinstance(model, MultiOutputClassifier):
        is_multioutput_style = True

    # Convert to np.array
    if is_multioutput_style:
        y_score_array = np.column_stack([proba[:, 1] for proba in y_score])
    else:
        y_score_array = np.array(y_score)
        
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(labels):
        fpr_dt, tpr_dt, _ = roc_curve(actual_array[:, i], y_score_array[:, i])
        roc_auc_dt_best = auc(fpr_dt, tpr_dt)
        all_auc.append(roc_auc_dt_best)
        
        plt.plot(
            fpr_dt,
            tpr_dt,
            lw=2,
            label=f'{class_name} (AUC = {roc_auc_dt_best:.4f})'
        )
    
    macro_auc = sum(all_auc) / len(all_auc)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type} ROC Curve ({target_name})\nOverall Macro-AUC: {macro_auc:.4f}')
    plt.legend()
    plt.show()


# =============================================================================
# Custom Estimators
# =============================================================================

class CustomMultiOutputEstimator:
    """
    Custom multi-output estimator that combines multiple single-label estimators.
    
    This is useful when using SMOTE with multi-label classification,
    as SMOTE needs to be applied separately for each label.
    
    Parameters
    ----------
    estimators : list
        List of trained estimators, one for each label.
    """
    
    def __init__(self, estimators):
        self.estimators = estimators
        
    def predict(self, X):
        """Generate predictions for each estimator and stack them."""
        predictions = [est.predict(X).reshape(-1, 1) for est in self.estimators]
        return np.hstack(predictions)
    
    def predict_proba(self, X):
        """Generate probability predictions for each estimator."""
        proba_list = []
        for est in self.estimators:
            proba = est.predict_proba(X)[:, 1].reshape(-1, 1)
            proba_list.append(proba)
            
        return np.hstack(proba_list)
