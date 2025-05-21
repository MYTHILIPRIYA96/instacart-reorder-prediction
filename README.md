#  Instacart Reorder Prediction (FFNN Model)

This project predicts whether a user will reorder a product based on their historical shopping behavior using a **feedforward neural network (FFNN)**. It involves data cleaning, feature engineering, preprocessing, and model training.

---

##  Project Structure

- **Raw Input**: Instacart dataset (CSV format)
- **Steps**:
  1. Merge and chunk large files (`orders`, `order_products__prior`, etc.)
  2. Generate `prior_data.csv` and `train_labels.csv`
  3. Feature engineering → `final_features.csv`
  4. Preprocessing & encoding
  5. Train FFNN in `model.ipynb`

---

##  Assumptions

- Files are stored locally and accessible via correct paths.
- IDs are consistent across datasets (e.g., `user_id`, `product_id`).
- Feature engineering assumes that prior orders are representative of future behavior.
- Dataset is too large to load at once → chunked processing used.
- Classification target is `reordered` (binary: 0 or 1).

---

##  Data Files

| File | Description |

| `orders.csv` | Metadata of all orders |
| `order_products__prior.csv` | Products from prior orders |
| `order_products__train.csv` | Training labels |
| `products.csv`, `aisles.csv`, `departments.csv` | Product metadata |
| `prior_data.csv` | Chunked merge of prior orders + metadata |
| `train_labels.csv` | Merged training data |
| `final_features.csv` | Engineered features for modeling |


## ⚙️ Preprocessing Steps

- **Merging & chunking**: Large CSVs are processed in chunks to avoid memory issues.
- **Feature Engineering**:
  - **User-level**: reorder ratio, total orders, days between orders.
  - **Product-level**: reorder rate, total times ordered, add-to-cart behavior.
  - **User-Product-level**: frequency of reorders, last order info, reorder gap.
- **Missing Values**:
  - Categorical → mode
  - Numerical → median
- **Encoding**:
  - Categorical IDs → category codes
  - Label encoding for product metadata
- **Scaling**:
  - Standardized numerical features (`StandardScaler`)


##  Model Architecture

A 3-layer Feedforward Neural Network (defined in `model.ipynb`) with:
- Dense layers + ReLU
- Batch normalization
- Dropout
- Output: Sigmoid activation (binary classification)

##  Evaluation Metrics

- Precision, Recall, F1 Score, Accuracy
- ROC-AUC
- Confusion Matrix
- Threshold tuning using Precision-Recall Curve
-  Save the trained model as `ffnn_model.h5`
-  
##  Inference and Prediction

- Load `ffnn_model.h5` in your application to predict reorder probabilities for new user-product pairs.
- Use the same preprocessing pipeline to prepare input features before prediction.

## Notes

- The dataset is large; processing is performed in chunks to manage memory efficiently.
- Threshold tuning is performed on precision-recall curve to maximize F1 score, 
  suitable for imbalanced classification.
- The model architecture uses dropout and batch normalization to improve 
   generalization.
## Reproducibility

- Global seed used: `42`
- Applies to:
  - `numpy`, `random`, `tensorflow` seeds
  - `train_test_split(random_state=42)`
  - `SMOTE(random_state=42)`
- Consistent use of `astype` and `fillna` to control types and NAs

## Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt

## License

This project is licensed under the MIT License.
