# Coffee Leaf Diseases Prediction

A machine learning project that classifies coffee leaf diseases using RGB and CMY color features with Decision Tree and K-Nearest Neighbors algorithms to identify four classes: Miner, Phoma, Rust, and No disease.

## Overview

This project is a reproduction of the coffee leaf disease classification method described in the research paper below, using machine learning techniques with RGB and CMY color features.

**Research Paper:**
- **Title**: Comparative Analysis of the Performance of the Decision Tree and K-Nearest Neighbors Methods in Classifying Coffee Leaf Diseases
- **Authors**: Adie Suryadi, Murhaban Murhaban, Rivansyah Suhendra
- **Published in**: Department of Information Technology, Teuku Umar University, Indonesia
- **URL**: [https://aptikom-journal.id/conferenceseries/article/view/649/272](https://aptikom-journal.id/conferenceseries/article/view/649/272)

**Dataset:**
- **Name**: Coffee Leaf Diseases
- **Source**: Kaggle
- **URL**: [https://www.kaggle.com/datasets/badasstechie/coffee-leaf-diseases/code](https://www.kaggle.com/datasets/badasstechie/coffee-leaf-diseases/code)

## Methodology

This implementation extracts color-based features from coffee leaf images:
- **RGB features**: Mean and standard deviation for each R, G, B channel (6 features)
- **CMY features**: Mean and standard deviation for each C, M, Y channel (6 features)
- **Total**: 12 color-based features per image

The features are then used to classify coffee leaves into four categories:
- **Miner**: Leaf miner disease
- **Phoma**: Phoma disease
- **Rust**: Coffee rust disease
- **No disease**: Healthy leaf

Machine learning algorithms used:
- Decision Tree
- K-Nearest Neighbors (KNN)

## Directory Structure

```
coffee-leaf-deseases-prediction-ml/
├── README.md                              # This file
├── requirements.txt                       # Required Python packages
├── coffee-leaf-deseases-prediction.ipynb  # Main Jupyter Notebook
├── best_model_dt.pkl                      # Best Decision Tree model
├── best_model_knn.pkl                     # Best KNN model
├── decision_tree_model.pkl                # Decision Tree model (paper params)
├── knn_model.pkl                          # KNN model (paper params)
└── dataset/
    ├── train_classes.csv                  # Training data labels
    ├── test_classes.csv                   # Test data labels
    └── coffee-leaf-diseases/
        ├── train/
        │   ├── images/                    # Training images
        │   └── masks/                     # Training masks
        └── test/
            ├── images/                    # Test images
            └── masks/                     # Test masks
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

2. Open `coffee-leaf-deseases-prediction.ipynb` in your browser

3. Execute cells in order:
   - **Preprocessing Data**: Data preprocessing and feature extraction
   - **Hyperparameter Tuning**: GridSearchCV for optimal hyperparameters
   - **Find the Best Model**: Model evaluation and comparison

### Execution Order

1. **Data Preprocessing Cells**: Feature extraction from images and data splitting
2. **Hyperparameter Tuning Cell**: 10-fold CV to find optimal parameters
3. **Model Evaluation Cells**:
   - Models using parameters from the paper
   - Models using optimal parameters found by CV
4. **Confusion Matrix Visualization**: Visual performance assessment

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

## Presentation Link
[link](https://www.canva.com/design/DAG7VIdj3JU/gUucrDZ4DdDSnQdzOftZZA/view?utm_content=DAG7VIdj3JU&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h13b915f254)

## License

This project is created for educational purposes. Please refer to the Kaggle dataset page for dataset licensing information.
