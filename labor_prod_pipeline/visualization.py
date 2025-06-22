"""
Filename: visualization.py
Author: Dang Nguyen
Description: A module that provides visualization tools to explore the relationship between
             GDP per person employed and various macroeconomic indicators. It includes functions
             for generating correlation heatmaps, standardized scatter plots, and time series
             comparisons across countries
"""

import re
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_countries_gdp_emp(df, countries):
    """
    Parameters: df - a DataFrame containing macroeconomic indicator data
                indicator - name of the indicator
    Returns: none
    Does: plots and displays the time series of GDP per person employed for selected countries
          in a line plot
    """
    gdp_emp_col_label = get_full_column_label(df, "GDP Per Person Employed")
    gdp_emp_title = re.sub(r"\s*\(.*?\)", "", gdp_emp_col_label).strip()

    country_df = df[df["Country"].isin(countries)]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize = (14, 12))
    sns.lineplot(data = country_df, x = "Year", y = gdp_emp_col_label, hue = "Country",
                 marker = "o").legend(title = None)
    plt.xlabel("Year", fontsize = 12)
    plt.ylabel(gdp_emp_col_label, fontsize = 12)
    plt.title(f"{gdp_emp_title} Over Time Between Countries", fontsize = 15)
    plt.show()