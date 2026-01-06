"""
Filename: preprocessing.py
Author: Dang Nguyen
Description: A module for fetching and preprocessing macroeconomic indicators from the World Bank
             and IMF APIs, interpolating missing values, and integrating the data into a single
             DataFrame for further analysis.
"""

import requests
import pandas as pd

WB_BASE_URL = "https://api.worldbank.org/v2/country/indicator/"
IMF_BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

WB_INDICATORS = {
    "GDP Per Person Employed": {"unit": "2021 PPP $", "id": "SL.GDP.PCAP.EM.KD"},
    "GDP Per Capita": {"unit": "current US$", "id": "NY.GDP.PCAP.CD"},
    "Labor Force Participation Rate": {"unit": "% of total population ages 15+",
                                       "id": "SL.TLF.CACT.ZS"},
    "Research And Development Expenditure": {"unit": "% of GDP", "id": "GB.XPD.RSDV.GD.ZS"},
    "Tertiary Education Enrollment": {"unit": "% gross", "id": "SE.TER.ENRR"},
}

IMF_INDICATORS = {
    "Unemployment Rate": {"unit": "% of total labor force", "id": "LUR"},
    "Government Expenditure": {"unit": "% of GDP", "id": "exp"}
}


class WBData:

    def __init__(self, base_url):
        """
        Parameters: base_url - base API endpoint for World Bank data
        Returns: none
        Does: initializes the WBData object with the given base URL
        """
        self.base_url = base_url

    def get_data(self, indicator_id):
        """
        Parameters: indicator_id - World Bank indicator ID
        Returns: a JSON response
        Does: fetches a specific indicator’s data using the World Bank API
        """
        initial_api_url = f"{self.base_url}{indicator_id}?format=json"
        initial_api_response = requests.get(initial_api_url)

        if initial_api_response.status_code != 200:
            print(f"Failed to retrieve {indicator_id} data. "
                  f"Status code: {initial_api_response.status_code}")
            return None

        initial_json_response = initial_api_response.json()
        total_values = initial_json_response[0]["total"]

        final_api_url = f"{self.base_url}{indicator_id}?format=json&per_page={total_values}"
        final_api_response = requests.get(final_api_url)

        if final_api_response.status_code != 200:
            print(f"Failed to retrieve {indicator_id} data. "
                  f"Status code: {final_api_response.status_code}")
            return None

        return final_api_response.json()

    def parse_data(self):
        """
        Parameters: none
        Returns: a DataFrame
        Does: extracts and formats data for all specified World Bank indicators into a DataFrame
        """
        data = {}
        for name, attributes in WB_INDICATORS.items():
            indicator_id = attributes["id"]
            json_response = self.get_data(indicator_id)
            indicator_data = json_response[1]

            for record in indicator_data:
                country = record["countryiso3code"]
                year = int(record["date"])
                value = record["value"]
                if (country, year) not in data:
                    data[(country, year)] = {"Country": country, "Year": int(year)}
                data[(country, year)][name] = value

        df = pd.DataFrame.from_dict(data, orient = "index").reset_index(drop = True)
        df.sort_values(by = ["Country", "Year"], inplace = True)
        df.replace("", pd.NA, inplace = True)
        df.dropna(subset = ["Country"], inplace = True)
        df.reset_index(drop = True, inplace = True)

        indicators = list(WB_INDICATORS.keys())
        df[indicators] = df.groupby("Country")[indicators].transform(
            lambda group: group.interpolate(method = "nearest"))

        return df


class IMFData:

    def __init__(self, base_url):
        """
        Parameters: base_url - base API endpoint for IMF data
        Returns: none
        Does: initializes the IMFData object with the given base URL
        """
        self.base_url = base_url

    def get_data(self, indicator_id):
        """
        Parameters: indicator_id - IMF indicator ID
        Returns: a JSON response
        Does: fetches a specific indicator’s data using the IMF API
        """
        api_url = f"{self.base_url}/{indicator_id}"
        api_response = requests.get(api_url)

        if api_response.status_code != 200:
            print(f"Failed to retrieve {indicator_id} data. "
                  f"Status code: {api_response.status_code}")
            return None

        return api_response.json()

    def parse_data(self):
        """
        Parameters: none
        Returns: a DataFrame
        Does: extracts and formats data for all specified IMF indicators into a DataFrame
        """
        data = {}
        for name, attributes in IMF_INDICATORS.items():
            indicator_id = attributes["id"]
            json_response = self.get_data(indicator_id)
            indicator_data = json_response["values"][indicator_id]

            for country, yearly_values in indicator_data.items():
                for year, value in yearly_values.items():
                    if (country, year) not in data:
                        data[(country, year)] = {"Country": country, "Year": int(year)}
                    data[(country, year)][name] = value

        df = pd.DataFrame.from_dict(data, orient = "index").reset_index(drop = True)
        df.sort_values(by = ["Country", "Year"], inplace = True)
        df.reset_index(drop = True, inplace = True)

        indicators = list(IMF_INDICATORS.keys())
        df[indicators] = df.groupby("Country")[indicators].transform(
            lambda group: group.interpolate(method = "nearest"))

        return df


class DataIntegrator:

    def __init__(self):
        """
        Parameters: none
        Returns: none
        Does: initializes instances of WBData and IMFData
        """
        self.wb_data = WBData(WB_BASE_URL)
        self.imf_data = IMFData(IMF_BASE_URL)

    def merge_dfs(self):
        """
        Parameters: none
        Returns: a DataFrame
        Does: merges the two datasets containing World Bank and IMF indicators
              using an outer join on Country and Year
        """
        wb_df = self.wb_data.parse_data()
        imf_df = self.imf_data.parse_data()

        merged_df = pd.merge(wb_df, imf_df, on = ["Country", "Year"], how = "outer")
        merged_df.sort_values(by = ["Country", "Year"], inplace = True)
        merged_df.reset_index(drop = True, inplace = True)

        return merged_df

    def filter_df(self):
        """
        Parameters: none
        Returns: a DataFrame
        Does: cleans the merged dataset by limiting records to the year 2024 or earlier,
              removing missing values, and applying column renaming and ordering
        """
        merged_df = self.merge_dfs()
        filtered_df = merged_df[merged_df["Year"] <= 2024].dropna().reset_index(drop = True)

        column_labels = {indicator: f"{indicator} ({WB_INDICATORS[indicator]['unit']})"
                         for indicator in WB_INDICATORS}
        column_labels.update({indicator: f"{indicator} ({IMF_INDICATORS[indicator]['unit']})"
                              for indicator in IMF_INDICATORS})
        filtered_df.rename(columns = column_labels, inplace = True)

        new_column_order = ["Country", "Year", "GDP Per Person Employed", "GDP Per Capita",
                            "Labor Force Participation Rate", "Unemployment Rate",
                            "Research And Development Expenditure", "Government Expenditure",
                            "Tertiary Education Enrollment"]

        ordered_column_labels = []
        for col in new_column_order:
            if col in filtered_df.columns:
                ordered_column_labels.append(col)
            elif col in column_labels:
                ordered_column_labels.append(column_labels[col])

        filtered_df = filtered_df[ordered_column_labels]

        return filtered_df