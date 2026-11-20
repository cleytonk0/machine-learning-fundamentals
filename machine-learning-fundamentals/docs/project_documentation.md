# Technical Documentation

## What Is Machine Learning?

Machine Learning (ML) is a field in which algorithms learn patterns from example data and use those patterns to make predictions or decisions on data they have not seen. In supervised learning, each training example contains input variables and a known target. The model learns a mapping from inputs to targets; evaluation on held-out examples helps estimate how well that mapping generalizes.

## Regression and Classification

**Regression** predicts a numerical, continuous value, such as a price. In this project, the target is the generated property price.

**Classification** predicts a category or class. Here, the classes are the three Iris species. Although a classifier may calculate probabilities internally, its predicted output is a class label.

## Predictors and Target

Predictor variables (often written as `X` or features) are the inputs supplied to a model. The target variable (`y`, label, or response) is the value the model is intended to predict.

- In the regression experiment, predictors are property size, bedroom count, and building age; the target is the synthetic price.
- In the classification experiment, predictors are four flower measurements; the target is the Iris species encoded as a class.

## Training and Test Data

The data is divided into a **training partition** and a **test partition**. The estimator learns from the training partition. The test partition is held out during fitting and is used to calculate metrics after prediction. This makes the evaluation more informative than measuring performance on the same examples used for learning, although a single split can still vary and is not a guarantee of production performance.

Both experiments use `train_test_split` with `test_size=0.2` and `random_state=42`, matching the original notebook. Thus, 20% of observations are held out and the split is repeatable. The classification split is not stratified in the original methodology.

## `fit()` and `predict()`

`fit(X_train, y_train)` estimates a model from training predictors and their known targets. The details depend on the estimator: a linear model estimates coefficients, while a decision tree selects feature splits.

`predict(X_test)` applies the fitted model to new predictor rows and returns predicted numeric values for regression or class labels for classification. The script compares these predictions with `y_test` to compute evaluation metrics.

## Linear Regression

Linear regression models a continuous target as a weighted sum of the input features plus an intercept:

`prediction = intercept + coefficient_1 * feature_1 + ... + coefficient_n * feature_n`

During fitting, `LinearRegression` chooses coefficients that minimize the sum of squared residuals in the training data. It is interpretable through its coefficients and is a reasonable model when a linear relationship is suitable. In this experiment, the synthetic target was itself generated from a linear formula with random noise, which is an important limitation when interpreting the score.

## Logistic Regression

Despite its name, logistic regression is a classification algorithm. It combines input features linearly and transforms the result into class probabilities. For multiclass problems, scikit-learn's implementation generalizes this process across classes; the predicted class is selected from the estimated probabilities. The Iris script uses `max_iter=200` as in the notebook.

Logistic regression often benefits from feature scaling, especially when feature magnitudes differ. The original notebook does not scale features, so the organized script preserves that methodology rather than silently changing it.

## Decision Trees

A decision tree repeatedly splits examples according to feature thresholds. Each split creates smaller groups, and predictions are made from the final leaf reached by an example. Trees can model nonlinear relationships and are relatively easy to visualize, but unrestricted trees may overfit. The maximum depth constrains complexity.

This project uses `max_depth=5` for the regression tree and `max_depth=3` for the classification tree, both with `random_state=42`, as in the notebook.

## Evaluation Metrics

### Regression

- **Mean Squared Error (MSE):** `mean((actual - predicted)²)`. Squaring makes larger errors count more. Lower values indicate smaller test errors. Its unit is the square of the target unit (here, squared currency units), so its magnitude should not be read as a direct currency error.
- **R² (coefficient of determination):** compares squared prediction error with the error from predicting the test-target mean. A value of 1 indicates perfect predictions, 0 corresponds to the mean baseline, and a negative value means worse performance than that baseline on this test set.

### Classification

For a given class:

- **Accuracy:** number of correct predictions divided by the total number of predictions.
- **Precision:** true positives divided by all examples predicted as that class. It addresses how trustworthy positive class predictions are.
- **Recall:** true positives divided by all actual examples of that class. It measures how many examples of the class were found.
- **F1-score:** harmonic mean of precision and recall; it is high when both are high.

The script reports macro-averaged precision, recall, and F1-score: it computes each class's score and averages the class scores equally. The classification report also provides per-class values plus macro and support-weighted averages. A **confusion matrix** counts predictions by actual class (rows) and predicted class (columns); off-diagonal counts are errors.

## Interpreting These Experiments

The original notebook recorded these regression results on its fixed test split:

| Model | MSE | R² |
|---|---:|---:|
| Linear Regression | 714,009,399.99 | 0.9667 |
| Decision Tree Regressor | 1,393,387,234.26 | 0.9350 |

On this synthetic test set, linear regression has lower MSE and higher R² than the depth-limited tree. This is consistent with the target being constructed from a linear formula, but it does not establish that the same model would be better for real property prices.

The notebook recorded 1.0000 accuracy for both Iris models. Its decision-tree classification report showed 1.00 precision, recall, and F1-score for all three classes, with 30 examples in total. The organized script also prints the logistic-regression report and both confusion matrices and calculates macro metrics, which were not explicitly printed in the notebook. These extra outputs are computed by running the same models on the same split; they are not claims about any production dataset.

## Limitations and Possible Improvements

- The housing observations and target are synthetic, based on a deliberately simple formula; they do not represent collected housing data.
- The metrics come from one fixed split and can be sensitive to how observations are partitioned. Cross-validation and repeated evaluation would provide a broader picture.
- The Iris dataset is small and commonly used for demonstrations. Perfect accuracy on one 30-row test partition is not evidence that the models will achieve perfect accuracy elsewhere.
- The original Iris split is not stratified, and the logistic-regression features are not scaled. Both can be studied in separate experiments while documenting any methodology change.
- Future work could add plots, baselines, hyperparameter experiments, automated tests, and carefully sourced real-world data with appropriate validation.
