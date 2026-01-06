"""
Filename: model.py
Author: Dang Nguyen
Description: A module for training and evaluating global and country-specific tree-based regression models
             to predict GDP per person employed using lagged macroeconomic indicators, including
             global pooled models, country-specific models, and global-to-country models.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

MODELS = {"Random Forest": RandomForestRegressor(random_state = 42),
          "Gradient Boosting": GradientBoostingRegressor(random_state = 42),
          "XGBoost": XGBRegressor(objective = "reg:squarederror", random_state = 42)}


def make_lagged_features(df, lags = 1):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                lags - number of years by which predictor variables are lagged
    Returns: a DataFrame
    Does: shifts all predictor variables by the specified number of years within each country,
          creates a lagged target feature (t-lags), and removes rows with missing lagged values
          to avoid look-ahead bias
    """
    df_lagged = df.sort_values(["Country", "Year"]).copy()

    target = "GDP Per Person Employed (2021 PPP $)"

    feature_cols = [c for c in df_lagged.columns if c not in ["Country", "Year", target]]
    df_lagged[feature_cols] = df_lagged.groupby("Country")[feature_cols].shift(lags)

    df_lagged[f"{target} (t-{lags})"] = df_lagged.groupby("Country")[target].shift(lags)
    df_lagged = df_lagged.dropna(subset = feature_cols + [f"{target} (t-{lags})", target])

    return df_lagged


def compute_perf_metrics(y_test_log, y_pred_log):
    """
    Parameters: y_test_log - true target values on the log1p scale
                y_pred_log - predicted target values on the log1p scale
    Returns: a dictionary
    Does: converts log1p-scale values back to the original scale and
          computes model performance metrics (MSE, R2, MAPE)
    """
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R2": r2, "MAPE (%)": mape}


def fit_models(X_train, y_train):
    """
    Parameters: X_train - training feature matrix
                y_train - training target vector on the log1p scale
    Returns: a dictionary
    Does: trains fresh copies of each model on the training set
    """
    trained_models = {}

    for name, model in MODELS.items():
        fresh_model = model.__class__(**model.get_params())
        fresh_model.fit(X_train, y_train)
        trained_models[name] = fresh_model

    return trained_models


def train_eval_global_models(df):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
    Returns: a dictionary
    Does: trains predefined models on pooled global data using lagged features
          to predict GDP per person employed and evaluates their performance on test data
          with a year-based (time-respecting) split
    """
    df = df.dropna()
    df = make_lagged_features(df, lags = 1)
    df = df.sort_values("Year").reset_index(drop = True)

    target = "GDP Per Person Employed (2021 PPP $)"

    years = sorted(df["Year"].unique())
    cutoff_year = years[int(0.8 * len(years))]

    train_df = df[df["Year"] <= cutoff_year]
    test_df = df[df["Year"] > cutoff_year]

    X_train = train_df.drop(columns = ["Country", "Year", target])
    y_train = np.log1p(train_df[target])

    X_test = test_df.drop(columns = ["Country", "Year", target])
    y_test = np.log1p(test_df[target])

    trained = fit_models(X_train, y_train)

    results = {}
    for name, model in trained.items():
        y_pred_log = model.predict(X_test)
        results[name] = compute_perf_metrics(y_test, y_pred_log)

    return results


def train_eval_country_models(df, min_years = 25):
    """
    Parameters: df - lagged macroeconomic DataFrame (must include Country, Year)
                min_years - minimum number of observations required per country after lagging
    Returns: a nested dictionary
    Does: trains separate predefined models for each eligible country data using lagged features
          to predict GDP per person employed and evaluates their performance on test data
          with a year-based (time-respecting) split
    """
    df = df.dropna()
    df = make_lagged_features(df, lags = 1)

    results = {}
    counts = df["Country"].value_counts()
    eligible_countries = counts[counts >= min_years].index.tolist()

    for country in eligible_countries:
        country_df = df[df["Country"] == country].sort_values("Year")

        X = country_df.drop(columns = ["Country", "Year", "GDP Per Person Employed (2021 PPP $)"])
        y = np.log1p(country_df["GDP Per Person Employed (2021 PPP $)"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

        trained = fit_models(X_train, y_train)

        results[country] = {}
        for name, model in trained.items():
            y_pred_log = model.predict(X_test)
            results[country][name] = {**compute_perf_metrics(y_test, y_pred_log), "N": len(country_df)}

    return results


def train_eval_global_to_country_models(df, min_years = 25):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                min_years - minimum number of observations required per country after lagging
    Returns: nested dictionary
    Does: trains predefined models on pooled global data once using lagged features and
          evaluates those same trained models on each eligible country's time-respecting
          holdout set to assess cross-country generalization performance
    """
    df = df.dropna()
    df = make_lagged_features(df, lags = 1)
    df = df.sort_values("Year").reset_index(drop = True)

    X_all = df.drop(columns = ["Country", "Year", "GDP Per Person Employed (2021 PPP $)"])
    y_all = np.log1p(df["GDP Per Person Employed (2021 PPP $)"])

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all,
                                                                        test_size = 0.2, shuffle = False)

    trained_global = fit_models(X_train_all, y_train_all)

    results = {}
    counts = df["Country"].value_counts()
    eligible_countries = counts[counts >= min_years].index.tolist()

    for country in eligible_countries:
        country_df = df[df["Country"] == country].sort_values("Year")

        X_c = country_df.drop(columns = ["Country", "Year", "GDP Per Person Employed (2021 PPP $)"])
        y_c = np.log1p(country_df["GDP Per Person Employed (2021 PPP $)"])

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c,
                                                                    test_size = 0.2, shuffle = False)

        results[country] = {}
        for name, model in trained_global.items():
            y_pred_log = model.predict(X_test_c)
            results[country][name] = {**compute_perf_metrics(y_test_c, y_pred_log), "N": len(country_df)}

    return results