"""
Filename: model.py
Author: Dang Nguyen
Description: A module for training and evaluating machine learning models to predict GDP per person employed.
             It includes functionality for feature importance analysis using tree-based regression models.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

MODELS = {"Random Forest": RandomForestRegressor(random_state = 42),
          "Gradient Boosting": GradientBoostingRegressor(random_state = 42)}


def feature_analysis(model, X_train):
    """
    Parameters: model - a trained tree-based model
                X_train - training features used to fit the model
    Returns: a DataFrame or None
    Does: computes feature importance scores of the input features based on the trained model
    """
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by = "Importance", ascending = False)
        return importance_df
    else:
        return None


def train_models(df):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
    Returns: a tuple
    Does: trains predefined models using standardized input features to predict
          the logarithm of GDP per person employed
    """
    df = df.drop(columns = ["Country", "Year", "Unemployment Rate (% of total labor force)",
                          "Government Expenditure (% of GDP)"])
    df = df.select_dtypes(include = ["number"]).dropna()

    x = df.drop(columns = ["GDP Per Person Employed (2021 PPP $)"])
    y = np.log1p(df["GDP Per Person Employed (2021 PPP $)"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    trained_models = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_train, X_test, y_train, y_test


def evaluate_models(models, X_test, y_test):
    """
    Parameters: models - a dictionary of trained models
                X_test - test set features
                y_test - test set target values
    Returns: a dictionary
    Does: evaluates trained models on test data and computes regression performance metrics
    """
    results = {}
    for name, model in models.items():
        y_pred = np.expm1(model.predict(X_test))
        y_test_original = np.expm1(y_test)

        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

        results[name] = {"MSE": mse, "R2": r2, "MAPE (%)": mape}

    return results