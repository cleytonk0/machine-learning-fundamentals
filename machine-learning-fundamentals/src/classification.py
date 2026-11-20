"""Compare classification models on the Scikit-learn Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    """Fit a model and return predictions, metrics, and the confusion matrix."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def run_experiment():
    """Load Iris, fit the models, and display metrics and reports."""
    iris = load_iris()
    predictors = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.2, random_state=42
    )

    models = {
        "LOGISTIC REGRESSION": LogisticRegression(max_iter=200),
        "DECISION TREE CLASSIFIER": DecisionTreeClassifier(
            max_depth=3, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        result = evaluate_classification_model(
            model, X_train, X_test, y_train, y_test
        )
        results[name] = result

        print(f"--- {name} ---")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision (macro): {result['precision_macro']:.4f}")
        print(f"Recall (macro): {result['recall_macro']:.4f}")
        print(f"F1-score (macro): {result['f1_macro']:.4f}")
        print("Confusion matrix (rows: actual class; columns: predicted class):")
        print(result["confusion_matrix"])
        print("Classification report:")
        print(
            classification_report(
                y_test,
                result["predictions"],
                target_names=iris.target_names,
                zero_division=0,
            )
        )

    return results


if __name__ == "__main__":
    run_experiment()
