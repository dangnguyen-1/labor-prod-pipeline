"""
Filename: visualization.py
Author: Dang Nguyen
Description: A module that provides visualization tools to explore the relationship between
             GDP per person employed and various macroeconomic indicators. It includes functions
             for generating correlation heatmaps, standardized scatter plots, time series
             comparisons across countries and regions, and grouped bar charts to compare
             model performance metrics between global and country-specific models
"""

import re
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_full_column_label(df, indicator_name):
    """
    Parameters: df - a DataFrame
                indicator_name - a partial or full name of the indicator
    Returns: a string or None
    Does: searches the DataFrame's columns for a label that contains the specified indicator name
    """
    for col in df.columns:
        if indicator_name in col:
            return col
    return None


def plot_gdp_emp_corr(df):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
    Returns: none
    Does: computes and visualizes the Spearman correlation matrix of macroeconomic indicators
          as a heatmap, highlighting their correlations with GDP per person employed
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize = (14, 12))

    corr_matrix = df.drop(columns = ["Country", "Year"]).corr(method = "spearman")

    sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5,
                annot_kws = {"fontfamily": "Times New Roman"})
    plt.xticks(rotation = 45, ha = "right", fontsize = 10)
    plt.yticks(rotation = 0, va = "center", fontsize = 10)
    plt.title("Correlation of Macroeconomic Indicators with GDP Per Person Employed")
    plt.tight_layout()
    plt.show()


def plot_indicator_vs_gdp_emp(df, indicator):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                indicator - name of the indicator
    Returns: none
    Does: visualizes the relationship between GDP per person employed with the selected indicator
          by standardizing both and displaying them in a scatter plot
    """
    indicator_col_label = get_full_column_label(df, indicator)
    gdp_emp_col_label = get_full_column_label(df, "GDP Per Person Employed")

    indicator_title = re.sub(r"\s*\(.*?\)", "", indicator_col_label).strip()
    gdp_emp_title = re.sub(r"\s*\(.*?\)", "", gdp_emp_col_label).strip()

    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[[indicator_col_label, gdp_emp_col_label]] = scaler.fit_transform(
        df[[indicator_col_label, gdp_emp_col_label]])

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize = (14, 12))
    sns.scatterplot(data = df_normalized, x = indicator_col_label, y = gdp_emp_col_label,
                    alpha = 0.7)
    sns.regplot(data = df_normalized, x = indicator_col_label, y = gdp_emp_col_label,
                scatter = False, color = "red")
    plt.xlabel(f"Standardized {indicator_title}", fontsize = 12)
    plt.ylabel(f"Standardized {gdp_emp_title}", fontsize = 12)
    plt.title(f"Standardized {gdp_emp_title} vs. {indicator_title}", fontsize = 15)
    plt.show()


def plot_countries_gdp_emp(df, countries, region_map):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                indicator - name of the indicator
                region_map - a dictionary that maps each country to a region name
    Returns: none
    Does: plots and displays the time series of GDP per person employed for selected countries
          with region-based coloring in a line plot
    """
    gdp_emp_col_label = get_full_column_label(df, "GDP Per Person Employed")
    gdp_emp_title = re.sub(r"\s*\(.*?\)", "", gdp_emp_col_label).strip()

    country_df = df[df["Country"].isin(countries)].copy()
    country_df["Region"] = country_df["Country"].map(region_map).fillna("Other")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize = (14, 12))

    ax = sns.lineplot(data = country_df, x = "Year", y = gdp_emp_col_label, hue = "Region",
                      units = "Country", estimator = None, linewidth = 2, legend = True)
    ax.legend(title = None)

    for c in countries:
        d = country_df[country_df["Country"] == c].sort_values("Year")
        if d.empty:
            continue
        last = d.iloc[-1]
        ax.text(last["Year"] + 0.2, last[gdp_emp_col_label], c, fontsize = 9, va = "center")

    ax.set_xlabel("Year", fontsize = 12)
    ax.set_ylabel(gdp_emp_col_label, fontsize = 12)
    ax.set_title(f"{gdp_emp_title} Over Time by Region", fontsize = 15)
    plt.show()


def plot_mape_comparison(country_results, global_results, country_order, title):
    """
    Parameters: country_results - a dictionary containing per-country model evaluation results
                global_results - a dictionary containing global pooled model evaluation results
                country_order = a list specifying the country order for the x-axis
                title - title of the chart
    Returns: none
    Does: plots and displays a grouped bar chart comparing MAPE between global pooled models
          and per-country models
    """
    countries = [c for c in country_order if c in country_results]
    labels = ["Global"] + countries

    rf_mape = ([global_results["Random Forest"]["MAPE (%)"]] +
               [country_results[c]["Random Forest"]["MAPE (%)"] for c in countries])

    gb_mape = ([global_results["Gradient Boosting"]["MAPE (%)"]] +
               [country_results[c]["Gradient Boosting"]["MAPE (%)"] for c in countries])

    xgb_mape = ([global_results["XGBoost"]["MAPE (%)"]] +
                [country_results[c]["XGBoost"]["MAPE (%)"] for c in countries])

    x = np.arange(len(labels))
    w = 0.25

    plt.figure()
    plt.bar(x - w, rf_mape, w, label = "Random Forest")
    plt.bar(x, gb_mape, w, label = "Gradient Boosting")
    plt.bar(x + w, xgb_mape, w, label = "XGBoost")

    plt.xticks(x, labels, fontsize = 10)
    plt.ylabel("MAPE (%)", fontsize = 10)
    plt.ylim(0, 40)
    plt.title(title, fontsize = 16)
    plt.legend()
    plt.show()