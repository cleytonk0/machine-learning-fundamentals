# Machine Learning Fundamentals: Regression and Classification

## Project Overview

This study project organizes two introductory machine learning experiments: predicting a continuous house-price target and classifying Iris flower species. It demonstrates a basic, repeatable workflow with scikit-learn models and holdout evaluation.

The house-price experiment uses **synthetic data generated in code**, not observed market data. The classification experiment uses the **Iris dataset included with scikit-learn**. Neither experiment is evidence of performance on real-world production data.

## Project Objectives

- Separate regression and classification workflows into readable Python scripts.
- Compare a linear model and a decision tree for regression.
- Compare logistic regression and a decision tree for multiclass classification.
- Practice train/test splitting, model fitting, prediction, and common evaluation metrics.
- Preserve the original executed notebook and its recorded outputs.

## Technologies and Libraries

- Python
- NumPy and pandas for synthetic data generation and tabular data
- scikit-learn for datasets, splitting, estimators, and evaluation
- Jupyter for the preserved notebook
- Matplotlib and Seaborn are included in the requested environment for notebook-based exploration; the current experiment scripts do not use them.

## Regression Experiment

`src/regression.py` creates 500 synthetic observations with three predictors: property size in square meters, number of bedrooms, and building age. The target price is generated from a linear formula with normally distributed noise. The script uses an 80/20 train/test split with `random_state=42`.

It compares:

- **Linear Regression (`LinearRegression`)**: estimates coefficients for a linear relationship between predictors and a continuous target.
- **Decision Tree Regressor (`DecisionTreeRegressor`)**: predicts by recursively splitting the predictor space into regions. The tree depth is limited to 5.

Both estimators are fitted on the training partition and evaluated on the held-out test partition using MSE and R².

## Classification Experiment

`src/classification.py` loads the 150-row Iris dataset supplied by scikit-learn. The predictors are sepal length and width and petal length and width; the target is one of three Iris species. It uses an 80/20 train/test split with `random_state=42`, matching the original notebook.

It compares:

- **Logistic Regression (`LogisticRegression`)**: estimates class probabilities using a logistic function and selects a class based on those probabilities. The iteration limit is 200.
- **Decision Tree Classifier (`DecisionTreeClassifier`)**: recursively partitions the feature space into class-predictive regions. The tree depth is limited to 3.

For each model, the script prints accuracy, macro-averaged precision, recall and F1-score, a confusion matrix, and a per-class classification report.

## Machine Learning Workflow

1. Generate or load a dataset.
2. Select predictor variables (`X`) and the target (`y`).
3. Split the data into training and test partitions.
4. Fit each estimator on the training data.
5. Predict targets for the held-out test data.
6. Calculate metrics and inspect errors using the confusion matrix (classification).

## Model Evaluation

- **Mean Squared Error (MSE)** is the average squared difference between actual and predicted values. Lower is better; it is expressed in squared target units.
- **R²** measures the proportion of target variance explained relative to predicting the test-set mean. A value closer to 1 is better for this comparison; it can be negative.
- **Accuracy** is the fraction of all predictions that are correct.
- **Precision** measures the fraction of predictions for a class that are correct.
- **Recall** measures the fraction of actual examples of a class that are correctly identified.
- **F1-score** is the harmonic mean of precision and recall.

The classification script reports macro averages, which calculate each class metric independently and give each class equal weight. Its classification report also includes per-class, macro, and support-weighted results. Confusion-matrix rows correspond to actual classes and columns to predicted classes.

## Results

The following regression metrics are the values recorded by the original notebook:

| Model | MSE | R² |
|---|---:|---:|
| Linear Regression | 714,009,399.99 | 0.9667 |
| Decision Tree Regressor | 1,393,387,234.26 | 0.9350 |

The original notebook recorded **1.0000 accuracy for both Iris classifiers** on its 30-example test partition. Its printed decision-tree report showed precision, recall, and F1-score of 1.00 for each of the three classes (supports: 10, 9, and 11). The organized script additionally calculates and prints these metrics and confusion matrices for both models, since the notebook did not print the logistic-regression report or either confusion matrix. The corresponding script output is preserved in `results/classification_results.txt`.

These are results from one fixed train/test split on a synthetic dataset and a small built-in benchmark dataset. They should not be interpreted as expected performance in production or on real housing data.

## How to Run

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and run the experiments:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/regression.py
python src/classification.py
```

Open the preserved notebook in Jupyter with:

```bash
jupyter notebook notebooks/models.ipynb
```

## Project Structure

```text
machine-learning-fundamentals/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── models.ipynb
├── src/
│   ├── regression.py
│   └── classification.py
├── results/
│   ├── regression_results.txt
│   └── classification_results.txt
└── docs/
    └── project_documentation.md
```

## Future Improvements

- Use cross-validation and repeated splits to assess variability rather than relying on one holdout split.
- Stratify the Iris split and report the split and class supports explicitly.
- Add plots for the synthetic data, predictions, and confusion matrices.
- Explore model parameters and compare against simple baselines.
- Apply a preprocessing pipeline (including feature scaling where appropriate) for logistic regression.
- Evaluate on a documented real-world housing dataset only if one is added, with appropriate validation and data provenance.
- Add automated tests for dataset shape, deterministic behavior, and metric outputs.
