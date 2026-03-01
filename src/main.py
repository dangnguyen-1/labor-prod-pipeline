"""
Filename: main.py
Description: A module that serves as the entry point for running the full data pipeline.
             It loads and filters macroeconomic indicator data from the World Bank and IMF APIs,
             visualizes key relationships, and trains and evaluates tree-based regression models under global,
             country-specific, and global-to-country settings using expanding-window walk-forward cross-validation.
"""

from src.preprocessing import DataIntegrator, WB_INDICATORS, IMF_INDICATORS
from src.visualization import plot_gdp_emp_corr, plot_indicator_vs_gdp_emp, plot_countries_gdp_emp, plot_mape_comparison
from src.model import train_eval_global_models, train_eval_country_models, train_eval_global_to_country_models

REGION_MAP = {
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "ARG": "South America", "URY": "South America", "COL": "South America",
    "ESP": "Europe", "FRA": "Europe", "ITA": "Europe",
    "ZAF": "Africa", "TUN": "Africa", "MAR": "Africa",
    "JPN": "Asia", "PAK": "Asia", "GEO": "Asia"}


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
    plot_countries_gdp_emp(indicators_df, list(REGION_MAP.keys()), REGION_MAP)

    # Train and evaluate global pooled models using all countries jointly
    # and country-specific models for countries with sufficient data
    global_results = train_eval_global_models(indicators_df, min_train_years = 20,
                                                               test_years = 3, step = 1)
    country_results = train_eval_country_models(indicators_df, min_years = 25,
                                                min_train_years = 15, test_years = 3, step = 1)

    # Train global pooled models and evaluate when applied to individual countries
    global_to_country_results = train_eval_global_to_country_models(indicators_df, min_years = 25,
                                                                    min_train_years = 20,
                                                                    test_years = 3, step = 1)

    country_order = sorted(country_results.keys())

    # Visualize and compare global vs country-specific models’ prediction accuracy
    # using mean walk-forward MAPE
    plot_mape_comparison(country_results, global_results, country_order = country_order,
                         title = "Mean Walk-Forward MAPE: Global vs Country-Specific Models")

    # Visualize and compare prediction accuracy for global models applied to individual countries
    # using mean walk-forward MAPE
    plot_mape_comparison(country_results = global_to_country_results, global_results = global_results, country_order = country_order,
                         title = "Mean Walk-Forward MAPE: Global Models Applied to Individual Countries")


if __name__ == "__main__":
    main()