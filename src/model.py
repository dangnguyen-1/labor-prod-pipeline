"""
Filename: model.py
Description: A module for training and evaluating global and country-specific tree-based regression models
             to predict GDP per person employed using lagged macroeconomic indicators. It implements expanding-window
             walk-forward cross-validation for global pooled, country-specific, and global-to-country evaluation settings.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
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


def expanding_year_folds(df, min_train_years = 20, test_years = 3, step = 1):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                min_train_years - minimum number of years used for the initial training period
                test_years - number of future years included in each test fold
                step - number of years the window advances after each fold
    Returns: a List
    Does: generates expanding-window walk-forward cross-validation folds
          based on unique years, where training uses past years and testing
          uses the next block of future years
    """
    years = sorted(df["Year"].unique())
    folds = []

    for i in range(min_train_years, len(years) - test_years + 1, step):
        train_years = years[:i]
        test_years_block = years[i:i + test_years]

        train_mask = df["Year"].isin(train_years)
        test_mask = df["Year"].isin(test_years_block)

        folds.append((train_mask, test_mask))

    return folds


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


def train_eval_global_models(df, min_train_years = 20, test_years = 3, step = 1):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                min_train_years - minimum number of years used for the initial training period
                test_years - number of future years included in each test fold
                step - number of years the window advances after each fold
    Returns: a dictionary
    Does: trains predefined models on pooled global data using lagged features and evaluates their performance
          using expanding-window walk-forward cross-validation, averaging out-of-sample metrics across folds
    """
    df = df.dropna()
    df = make_lagged_features(df, lags=1)
    df = df.sort_values(["Year", "Country"]).reset_index(drop=True)

    target = "GDP Per Person Employed (2021 PPP $)"

    X_all = df.drop(columns = ["Country", "Year", target])
    y_all = np.log1p(df[target])

    folds = expanding_year_folds(df, min_train_years = min_train_years,
                                 test_years = test_years, step = step)

    per_model_metrics = {name: [] for name in MODELS}

    for train_mask, test_mask in folds:
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        trained = fit_models(X_train, y_train)

        for name, model in trained.items():
            y_pred_log = model.predict(X_test)
            per_model_metrics[name].append(compute_perf_metrics(y_test, y_pred_log))

    results = {}
    for name, metrics_list in per_model_metrics.items():
        r2_vals = [m["R2"] for m in metrics_list]
        mape_vals = [m["MAPE (%)"] for m in metrics_list]

        results[name] = {
            "R2": float(np.mean(r2_vals)),
            "MAPE (%)": float(np.mean(mape_vals)),
            "Folds": len(metrics_list)
        }

    return results


def train_eval_country_models(df, min_years = 25, min_train_years = 15, test_years = 3, step = 1):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                min_train_years - minimum number of years used for the initial training period
                test_years - number of future years included in each test fold
                step - number of years the window advances after each fold
    Returns: a dictionary
    Does: trains separate predefined models for each eligible country using lagged features and evaluates performance
          with expanding-window walk-forward cross-validation within each country, averaging metrics across folds
    """
    df = df.dropna()
    df = make_lagged_features(df, lags = 1)

    target = "GDP Per Person Employed (2021 PPP $)"

    results = {}
    counts = df["Country"].value_counts()
    eligible_countries = counts[counts >= min_years].index.tolist()

    for country in eligible_countries:
        country_df = df[df["Country"] == country].sort_values("Year").reset_index(drop = True)

        X = country_df.drop(columns=["Country", "Year", target])
        y = np.log1p(country_df[target])

        folds = expanding_year_folds(country_df, min_train_years = min_train_years,
                                     test_years = test_years, step = step)

        per_model_metrics = {name: [] for name in MODELS}

        for train_mask, test_mask in folds:
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            trained = fit_models(X_train, y_train)

            for name, model in trained.items():
                y_pred_log = model.predict(X_test)
                per_model_metrics[name].append(compute_perf_metrics(y_test, y_pred_log))

        results[country] = {}
        for name, metrics_list in per_model_metrics.items():
            r2_vals = [m["R2"] for m in metrics_list]
            mape_vals = [m["MAPE (%)"] for m in metrics_list]

            results[country][name] = {
                "R2": float(np.mean(r2_vals)) if metrics_list else np.nan,
                "MAPE (%)": float(np.mean(mape_vals)) if metrics_list else np.nan,
                "Folds": len(metrics_list),
                "N": len(country_df)
            }

    return results


def train_eval_global_to_country_models(df, min_years = 25, min_train_years = 20, test_years = 3, step = 1):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                min_years - minimum number of usable observations required for a country
                min_train_years - minimum number of years used for the initial training period
                test_years - number of future years included in each test fold
                step - number of years the window advances after each fold
    Returns: a dictionary
    Does: trains predefined models on pooled global data within each expanding-window fold and
          evaluates those models on each eligible country’s future observations in the same fold
          to assess cross-country generalization performance, averaging results across folds
    """
    df = df.dropna()
    df = make_lagged_features(df, lags = 1)
    df = df.sort_values(["Year", "Country"]).reset_index(drop = True)

    target = "GDP Per Person Employed (2021 PPP $)"

    feature_cols = [c for c in df.columns if c not in ["Country", "Year", target]]
    counts = df["Country"].value_counts()
    eligible_countries = counts[counts >= min_years].index.tolist()

    folds = expanding_year_folds(df, min_train_years = min_train_years,
                                 test_years = test_years, step = step)

    per_country = {c: {m: [] for m in MODELS} for c in eligible_countries}

    for train_mask, test_mask in folds:
        train_df = df[train_mask]
        test_df = df[test_mask]
        X_train = train_df[feature_cols]
        y_train = np.log1p(train_df[target])

        trained_global = fit_models(X_train, y_train)

        for country in eligible_countries:
            country_test = test_df[test_df["Country"] == country]
            if country_test.empty:
                continue

            X_ct = country_test[feature_cols]
            y_ct = np.log1p(country_test[target])

            for name, model in trained_global.items():
                y_pred_log = model.predict(X_ct)
                per_country[country][name].append(compute_perf_metrics(y_ct, y_pred_log))

    results = {}
    for country in eligible_countries:
        results[country] = {}
        for model_name, ms in per_country[country].items():
            results[country][model_name] = {
                "R2": float(np.mean([m["R2"] for m in ms])) if ms else np.nan,
                "MAPE (%)": float(np.mean([m["MAPE (%)"] for m in ms])) if ms else np.nan,
                "Folds": len(ms),
                "N": int(counts[country])
            }

    return results