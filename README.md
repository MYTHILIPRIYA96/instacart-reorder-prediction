# Instacart-reorder-prediction
Deep learning-based reorder prediction using tabular features from Instacart orders dataset.
# Product Reorder Prediction using Deep Learning

## Project Overview

This project aims to predict whether a user will reorder a product based on their past order history and product information. We use a combination of feature engineering, resampling techniques to handle imbalanced data, and a fully connected feed-forward neural network (FFNN) to build a robust classification model.

Key components include:

Data preprocessing and feature engineering on large-scale order datasets.
User, product, and user-product aggregated features extraction.
Handling class imbalance using SMOTEENN (a hybrid over- and under-sampling method).
Training an improved FFNN model with batch normalization and dropout for regularization.
Model evaluation with precision-recall threshold tuning and multiple classification metrics.
Saving the trained model for future inference.

## Dataset

The dataset is sourced from Instacart orders and includes:

Orders information (`orders.csv`)
Prior ordered products (`order_products__prior.csv`)
Training labels (`order_products__train.csv`)
Product metadata (`products.csv`, `aisles.csv`, `departments.csv`)

## Project Structure

data_processing: Scripts for merging raw data files, chunk-wise processing, and feature aggregation.
feature_engineering: Scripts for preprocessing, missing value imputation, categorical encoding, and feature scaling.
train_model: Code for training the neural network model with resampling, early stopping, and learning rate adjustments.
evaluate_model: Scripts to evaluate model performance, plot confusion matrix, and print classification metrics.
ffnn_model.h5: Saved trained neural network model.
model.ipynb: full code
README.md: This file with project overview and instructions.



## Setup and Installation

1. **Clone the repository:**

    '''bash
    git clone https://github.com/mythilipriya/product-reorder-prediction.git
    cd product-reorder-prediction
    '''

2. **Create and activate a Python environment (recommended):**

    '''bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    '''

3. **Install required dependencies:**

    '''bash
    pip install -r requirements.txt
    '''

    *(If 'requirements.txt' is not provided, install manually:)*

    '''bash
    pip install pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn
    '''

## Usage Instructions

### 1. Data Processing

 Run data_processing to load raw CSV files, merge datasets in chunks, and generate aggregated feature files (`final_features.csv`) and training labels (`train_labels.csv`).

### 2. Feature Engineering

 Execute feature_engineering to perform:
   Missing value imputation
   Categorical encoding (Label Encoding and Category Codes)
   Numerical feature scaling with StandardScaler

### 3. Model Training and Evaluation

Use train_model to:
Load processed features and labels
Perform train/validation/test split with stratification
Apply SMOTEENN resampling to balance the training set
Train an improved feed-forward neural network with callbacks (early stopping, learning rate reduction)
Evaluate model on the test set with optimized thresholding
Save the trained model as `ffnn_model.h5`

### 4. Inference and Prediction

Load ffnn_model.h5 in your application to predict reorder probabilities for new user-product pairs.
Use the same preprocessing pipeline to prepare input features before prediction.

## Evaluation Metrics

 Accuracy
 Precision
 Recall
 F1 Score
 ROC-AUC
 Confusion Matrix visualization


## Notes

 The dataset is large; processing is performed in chunks to manage memory efficiently.
 Threshold tuning is performed on precision-recall curve to maximize F1 score, suitable for imbalanced classification.
 The model architecture uses dropout and batch normalization to improve generalization.

## Contact

For questions or contributions, please contact:

**Your Name** — your.email@example.com  
GitHub: [MYTHILIPRIYA96](https://github.MYTHILIPRIYA)

## License

This project is licensed under the MIT License.
