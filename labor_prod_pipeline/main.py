"""
Filename: main.py
Author: Dang Nguyen
Description: A module that acts as the entry point for running the data pipeline and controls
             the workflow for analyzing the relationship between macroeconomic indicators and
             GDP per person employed. It loads and filters data from the World Bank and IMF APIs,
             visualizes key patterns and correlations, trains machine learning models to predict
             GDP per person employed, and reports evaluation metrics for model performance.
"""

from preprocessing import DataIntegrator, WB_INDICATORS, IMF_INDICATORS
from visualization import plot_gdp_emp_corr, plot_indicator_vs_gdp_emp, plot_countries_gdp_emp
from model import train_models, evaluate_models


def main():
    # Load and filter macroeconomic indicator data from the World Bank and IMF into a DataFrame
    indicators_data = DataIntegrator()
    indicators_df = indicators_data.filter_df()

    # indicators_df.to_csv("filtered_data.csv", index = False)

    # Plot and display a Spearman correlation heatmap of all macroeconomic indicators
    plot_gdp_emp_corr(indicators_df)

    # Plot and display a scatter plot of standardized values for GDP per person employed
    # and each macroeconomic indicator
    for indicator in list(WB_INDICATORS.keys())[1:] + list(IMF_INDICATORS.keys()):
        plot_indicator_vs_gdp_emp(df = indicators_df, indicator = indicator)

    # Plot and display a line plot showing GDP per person employed trends over time for selected countries
    plot_countries_gdp_emp(indicators_df, ["USA", "CAN", "MEX",
                                           "ARG", "URY", "COL"
                                           "ESP", "FRA", "ITA",
                                           "ZAF", "TUN", "MAR"
                                           "JPN", "PAK", "GEO"])

    # Train regression models using macroeconomic indicators to predict GDP per person employed
    models, X_train, X_test, y_train, y_test, scaler = train_models(indicators_df)

    # Compute and report evaluation metrics for the trained models
    print("Model Performance:")
    print(evaluate_models(models, X_test, y_test))


if __name__ == "__main__":
    main()