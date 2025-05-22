#  Instacart Reorder Prediction (FFNN Model)

This project predicts whether a user will reorder a product based on their historical shopping behavior using a **feedforward neural network (FFNN)**. It involves data cleaning, merge data,feature engineering, preprocessing, and model training.


##  Project Structure

- **Raw Input**: Instacart dataset (CSV format)
- **Steps**:
  1. Merge and chunk large files ('orders', 'order_products__prior', etc.)
  2. Generate 'prior_data.csv' and 'train_labels.csv' 
  3. Feature engineering  'final_features.csv' 
  4. Preprocessing & encoding    
  5. Train FFNN in 'model.ipynb' 
  6. ffnn_model.h5 # Saved h5 model



##  Assumptions

- Files are stored locally and accessible via correct paths.
- IDs are consistent across datasets (e.g., 'user_id', 'product_id').
- Feature engineering assumes that prior orders are representative of future behavior.
- Dataset is too large to load at once → chunked processing used.
- Classification target is 'reordered' (binary: 0 or 1).

---

##  Data Files

| File | Description |

| 'orders.csv' | Metadata of all orders |
| 'order_products__prior.csv' | Products from prior orders |
| 'order_products__train.csv' | Training labels |
| 'products.csv', 'aisles.csv', 'departments.csv' | Product metadata |
| 'prior_data.csv' | Chunked merge of prior orders + metadata |
| 'train_labels.csv' | Merged training data |
| 'final_features.csv' | Engineered features for modeling |
| 'preprocessed_features.csv' | Cleaned dataset ready for modeling |


##  Preprocessing Steps

- **Merging & chunking**: Large CSVs are processed in chunks to avoid memory issues.
- **Feature Engineering**:
  - **User-level**: reorder ratio, total orders, days between orders.
  - **Product-level**: reorder rate, total times ordered, add-to-cart behavior.
  - **User-Product-level**: frequency of reorders, last order info, reorder gap.
- **Missing Values**:
  - Categorical : mode
  - Numerical : median
- **Encoding**:
  - Categorical IDs : category codes
  - Label encoding for product metadata
- **Scaling**:
  - Standardized numerical features ('StandardScaler')
- **Balance the Dataset**
  - Downsamples majority class.
  - Applies SMOTE to oversample the minority class in the training set.

##  Model Architecture

A 3-layer Feedforward Neural Network (defined in 'model.ipynb') with:
- Dense layers + ReLU
- Batch normalization
- Dropout
- Output: Sigmoid activation (binary classification)
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy

##  Evaluation Metrics

- Precision, Recall, F1 Score, Accuracy
- ROC-AUC
- Confusion Matrix is plotted for visual assessment.
- Threshold tuning using Precision-Recall Curve
- Save the trained model as 'ffnn_model.h5'
  
| Metric    | Description                                 |

| Accuracy  | Correct predictions / Total predictions     |
| Precision | TP / (TP + FP)                              |
| Recall    | TP / (TP + FN)                              |
| F1 Score  | Harmonic mean of Precision & Recall         |
| ROC-AUC   | Classifier's ability to rank positive cases |

**Confusion Matrix**
          |  Predicted 0	       |    Predicted 1        |
Actual 0	|  True Negatives (TN) |	False Positives (FP) |
Actual 1	|  False Negatives (FN)|	True Positives (TP)  |

True Positive (TP): Correctly predicted reorder

True Negative (TN): Correctly predicted non-reorder

False Positive (FP): Incorrectly predicted reorder

False Negative (FN): Missed reorder prediction

 - This helps evaluate model precision, recall, and overall performance in real-world 
    settings. 
##  Inference and Prediction

- Load 'ffnn_model.h5' in your application to predict reorder probabilities for new 
  user-product pairs.
- Use the same preprocessing pipeline to prepare input features before prediction.

## Notes

- The dataset is large; processing is performed in chunks to manage memory efficiently.
- Threshold tuning is performed on precision-recall curve to maximize F1 score, 
  suitable for imbalanced classification.
- The model architecture uses dropout and batch normalization to improve 
   generalization.
## Reproducibility

- Global seed used: '42'
- Applies to:
  - 'numpy', 'random', 'tensorflow' seeds
  - 'train_test_split(random_state=42)`\'
  - 'SMOTE(random_state=42)'
- Consistent use of 'astype' and 'fillna' to control types and NAs

## Requirements

Install all dependencies via:

'''bash
pip install -r requirements.txt
'''

## License

This project is licensed under the MIT License.
