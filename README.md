# Coffee Leaf Diseases Prediction

A machine learning project that classifies coffee leaf diseases using various feature extraction methods and classification algorithms to identify four classes: Miner, Phoma, Rust, and No disease.

## Overview

This project is a reproduction and extension of the coffee leaf disease classification method described in the research paper below, exploring multiple approaches including additional models, improved feature extraction, and multi-label classification.

**Research Paper:**
- **Title**: Comparative Analysis of the Performance of the Decision Tree and K-Nearest Neighbors Methods in Classifying Coffee Leaf Diseases
- **Authors**: Adie Suryadi, Murhaban Murhaban, Rivansyah Suhendra
- **Published in**: Department of Information Technology, Teuku Umar University, Indonesia
- **URL**: [https://aptikom-journal.id/conferenceseries/article/view/649/272](https://aptikom-journal.id/conferenceseries/article/view/649/272)

**Dataset:**
- **Name**: Coffee Leaf Diseases
- **Source**: Kaggle
- **URL**: [https://www.kaggle.com/datasets/badasstechie/coffee-leaf-diseases/code](https://www.kaggle.com/datasets/badasstechie/coffee-leaf-diseases/code)

## Notebooks

| Notebook | Description | Approach |
|----------|-------------|----------|
| `coffee-leaf-diseases-prediction.ipynb` | **Baseline** - Paper reproduction | RGB/CMY features, DT & KNN |
| `coffee-leaf-diseases-prediction-additional-models.ipynb` | Additional models | RGB/CMY features + Logistic Regression & Neural Network |
| `coffee-leaf-diseases-prediction-improved.ipynb` | Improved pipeline | Raw pixel + PCA + SMOTE |
| `coffee-leaf-diseases-prediction-multilabel.ipynb` | Multi-label classification | Raw pixel + PCA + MultiOutputClassifier |

## Methodology

### Feature Extraction Approaches

1. **RGB/CMY Color Features** (Baseline)
   - RGB features: Mean and standard deviation for each R, G, B channel (6 features)
   - CMY features: Mean and standard deviation for each C, M, Y channel (6 features)
   - Total: 12 color-based features per image

2. **Raw Pixel Data** (Improved/Multi-label)
   - Flattened pixel values from resized images
   - PCA for dimensionality reduction
   - SMOTE for class imbalance handling

### Classification Categories
- **Miner**: Leaf miner disease
- **Phoma**: Phoma disease
- **Rust**: Coffee rust disease
- **No disease**: Healthy leaf

### Machine Learning Algorithms
- Decision Tree
- K-Nearest Neighbors (KNN)
- Logistic Regression (Additional)
- Neural Network / MLP (Additional)

## Directory Structure

```
coffee-leaf-diseases-prediction-ml/
├── README.md                                            # This file
├── requirements.txt                                     # Required Python packages
├── utils.py                                             # Shared utility functions
├── coffee-leaf-diseases-prediction.ipynb                # Baseline notebook (paper reproduction)
├── coffee-leaf-diseases-prediction-additional-models.ipynb  # Additional models (LR, NN)
├── coffee-leaf-diseases-prediction-improved.ipynb       # Improved pipeline (PCA + SMOTE)
├── coffee-leaf-diseases-prediction-multilabel.ipynb     # Multi-label classification
├── models/
│   ├── best_model_dt.pkl                                # Best Decision Tree model
│   ├── best_model_knn.pkl                               # Best KNN model
│   ├── decision_tree_model.pkl                          # Decision Tree model (paper params)
│   └── knn_model.pkl                                    # KNN model (paper params)
└── dataset/
    ├── train_classes.csv                                # Training data labels
    ├── test_classes.csv                                 # Test data labels
    └── coffee-leaf-diseases/
        ├── train/
        │   ├── images/                                  # Training images
        │   └── masks/                                   # Training masks
        └── test/
            ├── images/                                  # Test images
            └── masks/                                   # Test masks
```

## Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Installation Steps

1. Clone the repository (or download)

2. Install required packages:
```bash
pip install -r requirements.txt
```
## How to Run

### Running with Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the desired notebook:
   - **Baseline**: `coffee-leaf-diseases-prediction.ipynb`
   - **Additional Models**: `coffee-leaf-diseases-prediction-additional-models.ipynb`
   - **Improved Pipeline**: `coffee-leaf-diseases-prediction-improved.ipynb`
   - **Multi-label**: `coffee-leaf-diseases-prediction-multilabel.ipynb`

3. Execute cells in order

### Notebook Descriptions

#### 1. Baseline (`coffee-leaf-diseases-prediction.ipynb`)
- Reproduces the research paper methodology
- Uses RGB/CMY color features (12 features)
- Decision Tree and KNN classifiers

#### 2. Additional Models (`coffee-leaf-diseases-prediction-additional-models.ipynb`)
- Extends baseline with Logistic Regression and Neural Network
- Same feature extraction as baseline
- Compares 4 models: DT, KNN, LR, NN

#### 3. Improved Pipeline (`coffee-leaf-diseases-prediction-improved.ipynb`)
- Raw pixel data instead of color features
- PCA for dimensionality reduction
- SMOTE for class imbalance handling

#### 4. Multi-label (`coffee-leaf-diseases-prediction-multilabel.ipynb`)
- Treats problem as multi-label classification
- Uses MultiOutputClassifier wrapper
- Per-label SMOTE for imbalance handling

## Results

After running the notebook, you will obtain:

- Accuracy, precision, recall, and F1-score for both Decision Tree and KNN models
- Confusion matrix heatmaps for each model
- Optimal hyperparameters discovered through GridSearchCV

## Dependencies

- **numpy**: Numerical computation and array operations
- **pillow**: Image loading and processing
- **pandas**: DataFrame operations and CSV handling
- **scikit-learn**: Machine learning models and evaluation metrics
- **seaborn**: Data visualization
- **matplotlib**: Plotting and graphics
- **imbalanced-learn**: SMOTE for class imbalance handling

## Presentation Link
[link](https://www.canva.com/design/DAG7VIdj3JU/gUucrDZ4DdDSnQdzOftZZA/view?utm_content=DAG7VIdj3JU&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h13b915f254)

## License

This project is created for educational purposes. Please refer to the Kaggle dataset page for dataset licensing information.
