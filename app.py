import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings

# # Set the background color of the app
# st.set_page_config(layout="wide", page_title="My Streamlit App", page_icon=":chart_with_upwards_trend:", initial_sidebar_state="collapsed")


warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)



merged_df = pd.read_csv("merged_data_final.csv")


def plot_arima_forecast(country_name, merged_df):
    
    country_X = merged_df.loc[merged_df['Country Name'] == country_name, [f'{year}_x' for year in range(1961, 2021)]]
    country_y = merged_df.loc[merged_df['Country Name'] == country_name, [f'{year}_y' for year in range(1961, 2021)]]

    
    country_X.dropna(inplace=True)
    country_y.dropna(inplace=True)

    
    y_row = country_y.iloc[0].values
    plt.figure(figsize=(10, 6))

    for i in range(7):
        X_row = country_X.iloc[i].values

        
        auto_model = auto_arima(y_row, exogenous=X_row, seasonal=False, trace=True, suppress_warnings=True)

        
        model = ARIMA(y_row, exog=X_row, order=auto_model.order)
        model_fit = model.fit()

        
        forecast_values = model_fit.forecast(steps=len(y_row), exog=X_row)

        
        forecast_mean = np.mean(forecast_values)
        actual_mean = np.mean(y_row)
        bias = actual_mean - forecast_mean
        corrected_forecast_values = forecast_values + bias

        
        indicator_code = merged_df.loc[merged_df['Country Name'] == country_name, 'Indicator Code_x'].iloc[i]

        
        plt.plot(corrected_forecast_values, label=indicator_code)

    
    plt.plot(y_row, label='Actual', linestyle='--', color='black')

    # Customize the plot
    plt.title(f'ARIMA Forecast for Food Index of {country_name} with Different Indicators')
    plt.xlabel('Time (Year)')
    plt.ylabel('Food Production Index')
    plt.legend()
    plt.ylim(bottom=0)  
    plt.xticks(np.arange(0, len(y_row), step=10), np.arange(1961, 2021, step=10))  # Shifting X-axis labels
    st.pyplot()  


def plot_arima_forecast2(country_name, indicator_code, merged_df):
    
    country_data = merged_df[(merged_df['Country Name'] == country_name) & (merged_df['Indicator Code_x'] == indicator_code)]
    
    
    if country_data.empty:
        st.write(f"No data found for country: {country_name} and indicator code: {indicator_code}")
        return
    
    
    X = country_data[[f'{year}_x' for year in range(1961, 2021)]].values
    y = country_data[[f'{year}_y' for year in range(1961, 2021)]].values.flatten()

    
    plt.figure(figsize=(10, 6))

    for i, X_row in enumerate(X):
        
        auto_model = auto_arima(y, exogenous=X_row.reshape(-1, 1), seasonal=False, trace=True, suppress_warnings=True)

        
        model = ARIMA(y, exog=X_row.reshape(-1, 1), order=auto_model.order)
        model_fit = model.fit()

        
        forecast_values = model_fit.forecast(steps=len(y), exog=X_row.reshape(-1, 1))

        
        forecast_mean = np.mean(forecast_values)
        actual_mean = np.mean(y)
        bias = actual_mean - forecast_mean
        corrected_forecast_values = forecast_values + bias

        
        plt.plot(corrected_forecast_values, label=indicator_code)

    
    plt.plot(y, label='Actual', linestyle='--', color='black')

    
    plt.title(f'ARIMA Forecast for {country_name} with Indicator {indicator_code}')
    plt.xlabel('Time')
    plt.ylabel('Food Production Index')
    plt.legend()
    st.pyplot()


def main():

    
    st.title("ARIMA Forecasting App")


    # Add navigation sidebar
    page = st.sidebar.radio("Go to", ("First Page", "Second Page"))

    if page == "First Page":
        # User input for country name
        country_name = st.text_input("Enter the country name:", "Aruba")

        # Button to trigger the ARIMA forecast plot
        if st.button("Plot ARIMA Forecast"):
            plot_arima_forecast(country_name, merged_df)

    elif page == "Second Page":
        st.title("Indicator wise")
        # User input for country name and indicator code
        country_name = st.text_input("Enter the country name:")
        indicator_code = st.text_input("Enter the indicator code:")

        # Button to trigger the ARIMA forecast plot for the second page
        if st.button("Plot ARIMA Forecast"):
            plot_arima_forecast2(country_name, indicator_code, merged_df)

# Execute the main function
if __name__ == "__main__":
    main()