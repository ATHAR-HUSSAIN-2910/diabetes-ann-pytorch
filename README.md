# Diabetes Prediction using Artificial Neural Network with PyTorch and Optuna

A portfolio-focused deep learning project that uses a **PyTorch Artificial Neural Network (ANN)** to predict diabetes from health-related, lifestyle, and demographic indicators.

The project follows a complete deep learning workflow, including **data exploration, preprocessing, leakage prevention, PyTorch Dataset/DataLoader creation, ANN development, manual training and validation, hyperparameter optimization with Optuna, early stopping, threshold optimization, model evaluation, artifact saving, and inference**.

---

## 📌 Project Overview

Diabetes prediction is a binary classification problem where the objective is to determine whether an individual belongs to the **diabetes** or **non-diabetes** class based on health-related and demographic indicators.

In this project, I built a feed-forward **Artificial Neural Network using PyTorch** and optimized its hyperparameters using **Optuna**.

The main goal was not simply to maximize accuracy, but to build a **clean, reproducible, and deployment-oriented deep learning workflow**.

### Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Train / Validation / Test Split
   ↓
Feature Scaling
   ↓
PyTorch Dataset
   ↓
PyTorch DataLoader
   ↓
Baseline ANN
   ↓
Optuna Hyperparameter Optimization
   ↓
Final ANN Training
   ↓
Early Stopping
   ↓
Threshold Optimization
   ↓
Final Test Evaluation
   ↓
Model + Scaler + Configuration Saving
   ↓
Inference Pipeline
```

---

## 🎯 Problem Statement

The objective of this project is to predict whether an individual has diabetes based on **21 health-related and demographic features**.

This is a **binary classification problem**.

| Target | Meaning     |
| ------ | ----------- |
| `0`    | No Diabetes |
| `1`    | Diabetes    |

The neural network produces a raw **logit**, which is converted into a probability using the sigmoid function.

A classification threshold is then applied to convert the probability into the final class prediction.

---

# 📊 Dataset

The project uses the **Diabetes Health Indicators Dataset** available on Kaggle.

**Dataset:** `alexteboul/diabetes-health-indicators-dataset`

**Selected file:**

```text
diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

### Dataset Summary

| Property            |             Value |
| ------------------- | ----------------: |
| Total observations  |            70,692 |
| Input features      |                21 |
| Target              | `Diabetes_binary` |
| Classification type |            Binary |
| Missing values      |                 0 |
| Duplicate rows      |             1,635 |
| Class 0             |            35,346 |
| Class 1             |            35,346 |

The dataset contains a balanced **50/50 target distribution**.

### Duplicate Handling

The dataset contains **1,635 duplicate rows**.

These rows were retained because identical feature values do not necessarily indicate incorrect observations. Without evidence that these records represent data errors, removing them could unnecessarily discard valid observations.

---

# 🧾 Features

The model uses the following 21 input features:

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

---

# 🔍 Exploratory Data Analysis

Several EDA steps were performed before model development.

### Target Distribution

The target variable is perfectly balanced:

```text
No Diabetes → 50%
Diabetes     → 50%
```

### Important Correlations

The strongest positive Pearson correlations with the target included:

| Feature                | Correlation |
| ---------------------- | ----------: |
| `GenHlth`              |       0.408 |
| `HighBP`               |       0.382 |
| `BMI`                  |       0.293 |
| `HighChol`             |       0.289 |
| `Age`                  |       0.279 |
| `DiffWalk`             |       0.273 |
| `PhysHlth`             |       0.213 |
| `HeartDiseaseorAttack` |       0.212 |

The strongest negative correlations included:

| Feature        | Correlation |
| -------------- | ----------: |
| `Income`       |      -0.224 |
| `Education`    |      -0.170 |
| `PhysActivity` |      -0.159 |

These correlations describe **linear relationships** with the target and are not treated as definitive feature importance for the neural network.

### EDA Included

* Target distribution analysis
* Feature distributions
* Correlation matrix
* Highly correlated feature pair analysis
* Boxplots across target classes
* Binary feature analysis
* Ordinal feature analysis

No pair of input features had an absolute Pearson correlation greater than **0.7**.

---

# 🧹 Data Preprocessing

## Train / Validation / Test Split

The dataset was divided into:

```text
70% → Training
15% → Validation
15% → Test
```

A fixed random seed of **42** was used.

Stratified splitting was applied to preserve the class distribution across the three datasets.

### Final Dataset Sizes

| Dataset    | Samples | Features |
| ---------- | ------: | -------: |
| Training   |  49,484 |       21 |
| Validation |  10,604 |       21 |
| Test       |  10,604 |       21 |

---

## Feature Scaling

`StandardScaler` from Scikit-learn was used for feature standardization.

The scaler was fitted **only on the training data**.

```text
FIT
Training data only

TRANSFORM
Training data
Validation data
Test data
```

This prevents information from the validation and test datasets from leaking into the preprocessing stage.

The fitted scaler is saved as:

```text
models/scaler.joblib
```

---

# 🧠 PyTorch Artificial Neural Network

The neural network was implemented using PyTorch's `nn.Module`.

The final architecture selected through Optuna is:

```text
Input
  ↓
21 Features
  ↓
Linear(21 → 96)
  ↓
ReLU
  ↓
Dropout(0.2)
  ↓
Linear(96 → 32)
  ↓
ReLU
  ↓
Dropout(0.2)
  ↓
Linear(32 → 1)
  ↓
Logit
```

### Why One Output?

This is a binary classification problem, so the network produces a **single logit**.

During evaluation, the probability is calculated using:

```python
probability = torch.sigmoid(logit)
```

The probability lies between:

```text
0 → 1
```

and is then converted into a class prediction using the selected threshold.

---

# ⚙️ Loss Function

The model uses:

```text
BCEWithLogitsLoss
```

The model outputs raw logits instead of applying a Sigmoid layer internally.

`BCEWithLogitsLoss` combines the sigmoid operation and binary cross-entropy calculation into a numerically stable loss function.

This is the recommended approach for binary classification with a single-output PyTorch network.

---

# 🚀 Baseline Model

Before performing hyperparameter optimization, a baseline ANN was trained to establish a reference point.

The baseline model achieved approximately:

```text
Validation Accuracy ≈ 75%
```

This demonstrated that the neural network was learning useful patterns from the dataset, while also providing a baseline for comparison with the optimized model.

---

# 🔬 Hyperparameter Optimization with Optuna

Optuna was used to search for a better neural network configuration.

The optimization objective was based on:

```text
Validation ROC-AUC
```

The **test set was not used during hyperparameter optimization**.

This keeps the test set independent for final model evaluation.

### Hyperparameters Tuned

The Optuna search included:

* Hidden layer size
* Dropout
* Learning rate
* Weight decay
* Batch size

---

## 🏆 Best Optuna Trial

The best trial achieved:

```text
Validation ROC-AUC: 0.8309
```

### Best Hyperparameters

| Hyperparameter | Best Value |
| -------------- | ---------: |
| Hidden Layer 1 |         96 |
| Hidden Layer 2 |         32 |
| Dropout        |        0.2 |
| Learning Rate  |   0.001212 |
| Weight Decay   |  0.0000030 |
| Batch Size     |        128 |

---

# ⏹️ Early Stopping

Early stopping was used during final model training to prevent unnecessary training after validation performance stopped improving.

The final training stopped after:

```text
16 epochs
```

The best validation loss was:

```text
0.5009
```

Early stopping helped prevent unnecessary training after validation performance stopped improving.

---

# 📈 Final Model Evaluation

The final model was evaluated on the **unseen test set**.

The classification threshold was initially set to `0.50` and was later optimized using the validation set.

The selected threshold was:

```text
0.40
```

## Final Test Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 74.16% |
| Precision | 68.53% |
| Recall    | 89.36% |
| F1-score  | 77.57% |
| ROC-AUC   | 82.98% |

The final test ROC-AUC was:

```text
0.8298
```

---

# 🎚️ Classification Threshold Analysis

A default classification threshold of `0.50` is not always optimal.

Several thresholds were evaluated using the **validation dataset**.

| Threshold | Precision | Recall |    F1 |
| --------: | --------: | -----: | ----: |
|      0.30 |     0.652 |  0.936 | 0.768 |
|      0.35 |     0.668 |  0.915 | 0.772 |
|      0.40 |     0.686 |  0.894 | 0.777 |
|      0.45 |     0.707 |  0.864 | 0.778 |
|      0.50 |     0.726 |  0.819 | 0.769 |
|      0.55 |     0.748 |  0.764 | 0.756 |
|      0.60 |     0.766 |  0.702 | 0.733 |
|      0.65 |     0.788 |  0.632 | 0.701 |
|      0.70 |     0.806 |  0.543 | 0.649 |

Among the tested thresholds, **0.45** produced the highest validation F1-score (0.778), with **0.40** close behind (0.777). The threshold of **0.40** was selected for final evaluation to favor recall while keeping the F1-score close to its peak.

### Why Use 0.40?

Lowering the threshold makes the model more likely to classify an observation as positive.

This resulted in:

* Higher recall
* Higher F1-score
* Lower precision
* Slightly lower accuracy

This trade-off is important in classification problems where missing positive cases may be more costly than generating additional false positives.

---

# 📊 Test Performance: Threshold Comparison

The final model was evaluated using both the default threshold of `0.50` and the selected threshold of `0.40`.

| Metric    | Threshold 0.50 | Threshold 0.40 |
| --------- | -------------: | -------------: |
| Accuracy  |         75.23% |         74.16% |
| Precision |         72.39% |         68.53% |
| Recall    |         81.55% |         89.36% |
| F1-score  |         76.70% |         77.57% |
| ROC-AUC   |         82.98% |         82.98% |

The threshold of `0.40` increased recall from **81.55% to 89.36%** and improved F1-score from **76.70% to 77.57%**, while slightly reducing accuracy and precision.

The threshold was selected using the validation data **before** being applied to the test set.

---

# 📌 Confusion Matrix

The final confusion matrix is available at:

```text
results/figures/confusion_matrix.png
```

The confusion matrix provides information about:

* True Negatives
* False Positives
* False Negatives
* True Positives

The relatively high recall at the selected threshold means that the model identifies a larger proportion of the positive class, at the cost of additional false positives.

---

# 📉 ROC Curve

The final ROC curve is available at:

```text
results/figures/roc_curve.png
```

The final test ROC-AUC is:

```text
0.8298
```

ROC-AUC evaluates how well the model separates the two classes across different classification thresholds.

---

# 💾 Model Artifacts

The trained model and preprocessing components are stored in the `models/` directory.

```text
models/
├── diabetes_ann_model.pth
├── scaler.joblib
└── model_config.json
```

### `diabetes_ann_model.pth`

Contains the trained PyTorch model weights.

### `scaler.joblib`

Contains the fitted `StandardScaler` used during preprocessing.

### `model_config.json`

Contains the model configuration, including:

* Input size
* Hidden layer sizes
* Dropout
* Learning rate
* Weight decay
* Batch size
* Selected classification threshold

---

# 🔮 Inference Pipeline

The project includes an inference function that follows the same preprocessing and prediction steps used during model development.

```text
New Input
   ↓
Validate / Format Input
   ↓
Apply Saved Scaler
   ↓
Convert to PyTorch Tensor
   ↓
Model Inference
   ↓
Sigmoid
   ↓
Probability
   ↓
Threshold = 0.40
   ↓
Prediction
```

The inference pipeline returns:

* Predicted probability
* Predicted class

This allows the saved model to be reused on new observations without retraining the network.

---

# 📁 Project Structure

```text
diabetes-ann-pytorch/
│
├── data/
│
├── notebooks/
│   └── diabetes_ann_pytorch.ipynb
│
├── models/
│   ├── diabetes_ann_model.pth
│   ├── scaler.joblib
│   └── model_config.json
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── training_validation_loss.png
│   │
│   └── metrics/
│       ├── classification_report.txt
│       ├── final_metrics.csv
│       └── threshold_comparison.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Programming language           |
| PyTorch          | Neural network implementation  |
| NumPy            | Numerical operations           |
| Pandas           | Data manipulation              |
| Scikit-learn     | Preprocessing and evaluation   |
| Matplotlib       | Data visualization             |
| Seaborn          | Statistical visualization      |
| Optuna           | Hyperparameter optimization    |
| KaggleHub        | Programmatic dataset download  |
| Joblib           | Saving preprocessing artifacts |
| Jupyter Notebook | Development environment        |

---

# 💻 How to Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/ATHAR-HUSSAIN-2910/diabetes-ann-pytorch.git
cd diabetes-ann-pytorch
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Open the Notebook

Open:

```text
notebooks/diabetes_ann_pytorch.ipynb
```

Run the notebook from top to bottom.

The dataset is downloaded programmatically using **KaggleHub**.

---

# ☁️ Google Colab

The notebook is designed to be portable between:

* VS Code
* Jupyter Notebook
* Google Colab

The project does **not require a GPU**.

The PyTorch device is selected automatically:

```python
torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Therefore, the project can run on CPU and can also take advantage of CUDA when a compatible GPU is available.

---

# 🔐 Reproducibility

Several practices were used to make the project reproducible:

* Fixed random seed: `42`
* Stratified train/validation/test split
* Training-only scaler fitting
* Saved preprocessing scaler
* Saved model weights
* Saved model configuration
* Saved classification threshold
* Programmatic dataset download
* Hyperparameter optimization performed using validation data
* Test set reserved for final evaluation

---

# 📚 Key Learnings

This project provided practical experience with:

* Building neural networks using PyTorch
* Creating custom `Dataset` classes
* Using `DataLoader`
* Writing manual training loops
* Writing validation loops
* Using `model.train()`
* Using `model.eval()`
* Working with logits and probabilities
* Using `BCEWithLogitsLoss`
* Preventing preprocessing leakage
* Hyperparameter optimization with Optuna
* Early stopping
* Classification threshold optimization
* Evaluating binary classifiers
* Saving and loading model artifacts
* Building a reusable inference pipeline

---

# 🚧 Future Improvements

Possible future improvements include:

* Comparing the ANN with traditional machine learning models
* More extensive Optuna optimization
* Probability calibration
* More detailed threshold analysis
* Experiment tracking
* Model explainability
* API deployment
* Interactive web application
* Containerization with Docker

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

The model is **not a medical diagnostic system** and should not be used to diagnose diabetes or make medical decisions.

The reported performance is specific to the dataset, preprocessing pipeline, model architecture, hyperparameters, threshold selection, and evaluation procedure used in this project.

---

# 👨‍💻 Author

**Athar Hussain**

Aspiring Data Scientist & Machine Learning Engineer

---

# ⭐ Project Highlights

* Built a complete Artificial Neural Network using **PyTorch**
* Used **Optuna** for hyperparameter optimization
* Implemented a custom **PyTorch Dataset and DataLoader**
* Applied **leakage-safe feature scaling**
* Used **early stopping**
* Optimized the classification threshold using validation data
* Achieved **0.8298 ROC-AUC** on the final test set
* Achieved **89.36% recall** at the selected threshold
* Achieved **77.57% F1-score** at the selected threshold
* Built a reusable inference pipeline
* Saved model weights, preprocessing, and configuration artifacts

---

## 📌 Final Takeaway

The most important outcome of this project is not simply the accuracy score.

The project demonstrates an end-to-end deep learning workflow where:

```text
Data
 ↓
EDA
 ↓
Leakage-Safe Preprocessing
 ↓
PyTorch Dataset & DataLoader
 ↓
ANN
 ↓
Optuna Optimization
 ↓
Early Stopping
 ↓
Threshold Optimization
 ↓
Evaluation
 ↓
Model Saving
 ↓
Inference
```

The final model achieved:

```text
ROC-AUC : 0.8298
Recall  : 89.36%
F1      : 77.57%
```

The selected threshold of **0.40** prioritizes higher recall while maintaining a reasonable balance between precision and recall.

This makes the project a practical demonstration of building, optimizing, evaluating, and preparing a **PyTorch deep learning model for reuse**.
