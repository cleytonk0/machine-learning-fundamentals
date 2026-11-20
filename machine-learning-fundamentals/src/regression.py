"""Compare regression models on a synthetic house-price dataset."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def create_dataset(n_samples=500):
    """Create the synthetic data used in the original notebook."""
    np.random.seed(42)

    size_sqm = np.random.normal(150, 50, n_samples).clip(50, 400)
    bedrooms = np.random.randint(1, 5, n_samples)
    age_years = np.random.uniform(0, 30, n_samples)
    price_brl = (
        (size_sqm * 3000)
        + (bedrooms * 15000)
        - (age_years * 1000)
        + np.random.normal(0, 25000, n_samples)
    )

    return pd.DataFrame(
        {
            "Size_sqm": size_sqm,
            "Bedrooms": bedrooms,
            "Age_years": age_years,
            "Price_BRL": price_brl,
        }
    )


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    """Fit a model, predict test values, and calculate evaluation metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def run_experiment():
    """Run and display the comparison between the two regression models."""
    data = create_dataset()
    predictors = data[["Size_sqm", "Bedrooms", "Age_years"]]
    target = data["Price_BRL"]

    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.2, random_state=42
    )

    models = {
        "LINEAR REGRESSION": LinearRegression(),
        "DECISION TREE REGRESSOR": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate_regression_model(
            model, X_train, X_test, y_train, y_test
        )
        print(f"--- {name} ---")
        print(f"MSE: {results[name]['mse']:.2f}")
        print(f"R²: {results[name]['r2']:.4f}\n")

    return results


if __name__ == "__main__":
    run_experiment()
