import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar


def fill_missing_months(data, month_col, value_col):
    """
    Fill missing months in a DataFrame with zero values.

    This function takes a DataFrame `data` containing monthly data, along with the column names for months (`month_col`)
    and corresponding values (`value_col`). It creates a DataFrame containing all twelve months, then merges it with the
    input DataFrame to ensure that all months are present. Missing values in the `value_col` column are filled with zeros.

    Parameters:
        data (DataFrame): The DataFrame containing the monthly data.
        month_col (str): The column name for the months.
        value_col (str): The column name for the corresponding values.

    Returns:
        DataFrame: The DataFrame with missing months filled, containing all twelve months with corresponding values.
    """
    all_months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']

    full_months_df = pd.DataFrame(all_months, columns=[month_col])

    merged_df = pd.merge(full_months_df, data, on=month_col, how='left')

    merged_df[value_col].fillna(0, inplace=True)

    return merged_df


def calculate_monthly_totals(data, location, equipment, year, col_name):
    """
    Calculate the total power consumption on a monthly basis for a specific location, equipment, and year.

    This function takes in a DataFrame `data` containing power consumption data, and calculates the total power consumption
    on a monthly basis for the specified `location`, `equipment`, and `year`. The power consumption data should include a
    column specified by `col_name` containing timestamps.

    Parameters:
        data (DataFrame): The DataFrame containing power consumption data.
        location (str): Identifier for the location where the data was recorded.
        equipment (str): Identifier for the equipment being monitored.
        year (int): The year for which the data is being analyzed.
        col_name (str): The column name containing timestamps.

    Returns:
        DataFrame: DataFrame with monthly total power consumption data, with columns for month and total power consumed.
    """
    # Ensure the 'timestamps' column is in datetime format
    data[col_name] = pd.to_datetime(data[col_name])

    # Extract year and month from the datetime
    data['year'] = data[col_name].dt.year
    data['month'] = data[col_name].dt.month

    # Filter the data based on the input location, equipment, and year
    filtered_data = data[(data['location'] == location) & (data['equipment'] == equipment) & (data['year'] == year)]

    # Group by year and month, then calculate the sum of used power
    result = filtered_data.groupby(['month'])['used_power'].sum().reset_index()

    return result

def convert_numeric_month_to_name(df, column_name):
    """
    Convert numeric month values to month names.

    This function maps the numeric month values in the specified column of the DataFrame `df` to their corresponding
    month names using the calendar module.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column containing numeric month values to be converted.

    Returns:
        DataFrame: The DataFrame with the specified column updated to contain month names instead of numeric values.
    """
    # Map the numeric months to month names using the calendar module
    df[column_name] = df[column_name].apply(lambda x: calendar.month_name[x])

    return df

def calculate_monthly_means(data, location, equipment, year, col_name):
    """
    Calculate the monthly mean temperature for a specific location and equipment in a given year.

    This function takes a DataFrame `data` containing temperature data with a timestamp column specified by `col_name`.
    It ensures that the timestamp column is in datetime format, extracts the year and month from the datetime,
    filters the data based on the input `location`, `equipment`, and `year`, and then calculates the mean temperature
    for each month.

    Parameters:
        data (DataFrame): DataFrame containing temperature data with a timestamp column.
        location (str): Identifier for the location where the temperature data was recorded.
        equipment (str): Identifier for the equipment for which temperature data is being analyzed.
        year (int): The year for which the temperature data is being analyzed.
        col_name (str): Name of the column containing timestamps.

    Returns:
        DataFrame: DataFrame containing the monthly mean temperature data for the specified location, equipment, and year.
    """
    # Ensure the 'timestamps' column is in datetime format
    data[col_name] = pd.to_datetime(data[col_name])

    # Extract year and month from the datetime
    data['year'] = data[col_name].dt.year
    data['month'] = data[col_name].dt.month

    # Filter the data based on the input location, equipment, and year
    filtered_data = data[(data['location'] == location) & 
                         (data['equipment'] == equipment) & 
                         (data['year'] == year)]

    # Group by year and month, then calculate the mean temperature
    result = filtered_data.groupby(['month'])['temperature'].mean().reset_index()

    return result

def merge_dataframes(df1, df2, col_name):
    """
    Merge two dataframes based on a common column.

    Parameters:
        df1 (DataFrame): DataFrame containing power consumption data.
        df2 (DataFrame): DataFrame containing temperature data.
        col_name (str): Column name to merge the dataframes on. Default is 'month'.

    Returns:
        DataFrame: Merged DataFrame.
    """
    # Merge the two DataFrames on merge_column
    merged_df = pd.merge(df1, df2, on=col_name)

    return merged_df

def calculate_daily_totals(data, location, equipment, year, col_name):
    """
    Calculate the daily total power consumption for a specific location, equipment, and year.

    This function takes a DataFrame `data` containing power consumption data and extracts daily total power consumption
    for the specified `location`, `equipment`, and `year` based on the provided column name `col_name` representing timestamps.

    Parameters:
        data (DataFrame): The DataFrame containing power consumption data.
        location (str): Identifier for the location where the data was recorded.
        equipment (str): Identifier for the equipment being monitored.
        year (int): The year for which the data is being analyzed.
        col_name (str): Column name representing timestamps.

    Returns:
        DataFrame: A DataFrame containing the daily total power consumption for the specified location, equipment, and year.
                   The DataFrame has columns for month, day, and the sum of power consumption.
    """
    # Ensure the timestamps column is in datetime format
    data[col_name] = pd.to_datetime(data[col_name])

    # Extract year, month, and day from the timestamps
    data['year'] = data[col_name].dt.year
    data['month'] = data[col_name].dt.month
    data['day'] = data[col_name].dt.day

    # Filter the data based on the input location, equipment, and year
    filtered_data = data[(data['location'] == location) & (data['equipment'] == equipment) & (data['year'] == year)]

    # Group by month and day, then calculate the sum of used power
    result = filtered_data.groupby(['month', 'day'])['used_power'].sum().reset_index()

    return result


def calculate_daily_means(data, location, equipment, year, col_name):
    """
    Calculate the daily mean temperature for a specific location and equipment in a given year.

    This function takes a DataFrame `data` containing temperature data with a timestamp column specified by `col_name`.
    It first converts the timestamp column to datetime format and extracts the year, month, and day information.
    Then, it filters the data based on the provided `location`, `equipment`, and `year`.
    Next, it calculates the daily mean temperature by grouping the filtered data by month and day and computing the mean temperature.
    The result is returned as a DataFrame.

    Parameters:
        data (DataFrame): The DataFrame containing temperature data.
        location (str): Identifier for the location.
        equipment (str): Identifier for the equipment.
        year (int): The year for which the data is being analyzed.
        col_name (str): The name of the timestamp column.

    Returns:
        DataFrame: A DataFrame containing the daily mean temperature for the specified location, equipment, and year.
    """
    # Ensure the timestamp column is in datetime format
    data[col_name] = pd.to_datetime(data[col_name])

    # Extract year, month, and day from the datetime
    data['year'] = data[col_name].dt.year
    data['month'] = data[col_name].dt.month
    data['day'] = data[col_name].dt.day

    # Filter the data based on the input location, equipment, and year
    filtered_data = data[(data['location'] == location) & (data['equipment'] == equipment) & (data['year'] == year)]

    # Group by month and day, then calculate the mean temperature
    result = filtered_data.groupby(['month', 'day'])['temperature'].mean().reset_index()

    return result

def plot_components(df, year, equipment, month, trend, seasonal, residual):
    """
    Plot power usage components (Original, Residual, Trend, and Seasonal) on a daily basis.

    Parameters:
        df (DataFrame): DataFrame containing power consumption data.
        year (int): Year for which to plot the data.
        equipment (str): Equipment identifier.
        month (int): Month for which to plot the data.
        trend (Series): Trend component of the time series.
        seasonal (Series): Seasonal component of the time series.
        residual (Series): Residual component of the time series.

    Returns:
        None
    """
    # Filter the DataFrame for the specified year, equipment, and month
    df_filtered = df[(df['year'] == year) & (df['equipment'] == equipment) & (df['month'] == month)]

    # Group the data by day and calculate the total power usage for each day
    daily_power = df_filtered.groupby('day')['used_power'].sum()

    # Create an array of days for the x-axis
    days = np.arange(1, len(daily_power) + 1)

    # Plot the power usage components on a daily basis
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 12))

    # Plot the original power usage
    axes[0].plot(days, daily_power, linestyle='-')
    axes[0].set_title(f'Original Power Usage on a Daily Basis for {equipment} in {year}-{month}')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Power Consumption')

    # Plot the seasonal component
    axes[1].plot(days, seasonal, color='green')
    axes[1].set_title('Seasonal Component')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Value')

    # Plot the trend component
    axes[2].plot(days, trend, color='blue')
    axes[2].set_title('Trend Component')
    axes[2].set_xlabel('Day')
    axes[2].set_ylabel('Value')

    # Plot the residual component
    axes[3].plot(days, residual, color='red')
    axes[3].set_title('Residual Component')
    axes[3].set_xlabel('Day')
    axes[3].set_ylabel('Value')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def combine_components(trend, seasonal, residual):
    """
    Combine the trend, seasonal, and residual components to reconstruct the original time series data.

    Parameters:
        trend (Series): Trend component of the time series.
        seasonal (Series): Seasonal component of the time series.
        residual (Series): Residual component of the time series.

    Returns:
        reconstructed_time_series (Series): Reconstructed time series data.
    """
    # Create a DataFrame with daily index
    index = pd.date_range('2022-04-01', periods=len(trend), freq='D')

    # Combine components into a DataFrame
    components_df = pd.DataFrame({'Trend': trend.values, 'Seasonal': seasonal.values, 'Residual': residual.values}, index=index)

    # Reconstruct the original time series data
    reconstructed_time_series = components_df['Trend'] + components_df['Seasonal'] + components_df['Residual']

    return reconstructed_time_series

def plot_combined_components(trend, seasonal, residual):
    """
    Plot the combined components along with the original time series data.

    Parameters:
        trend (Series): Trend component of the time series.
        seasonal (Series): Seasonal component of the time series.
        residual (Series): Residual component of the time series.

    Returns:
        None
    """
    # Create a time index from 0 to n days
    days = np.arange(len(trend))

    # Plot the combined components
    plt.figure(figsize=(24, 8))
    plt.plot(days, trend + seasonal + residual, label='Combined Components', color='blue')

    # Set x-axis ticks every 4 days from 0 to n
    plt.xticks(np.arange(0, len(trend) + 1, 4))

    # Add labels and legend
    plt.xlabel('Days')
    plt.ylabel('Power Consumed')
    plt.title('Combined Components Plot')
    plt.legend()

    # Show plot
    plt.show()

def plot_daily_power_usage(df, location, equipment, year, month):
    """
    Plot the total daily power usage for a particular day, month, year, location, and equipment.

    Parameters:
        df (DataFrame): DataFrame containing power consumption data.
        location (str): Location identifier.
        equipment (str): Equipment identifier.
        year (int): Year.
        month (int): Month.

    Returns:
        None
    """
    # Filter the DataFrame for the selected location, equipment, year, and month
    df_filtered = df[(df['location'] == location) & (df['equipment'] == equipment) &
                     (df['year'] == year) & (df['month'] == month)]

    # Aggregate the data over days and calculate total daily power usage
    daily_power_usage = df_filtered.groupby('day')['used_power'].sum() / 1000  # Convert to k(1k=1000)

    # Plot the total daily power usage
    plt.figure(figsize=(24, 8))
    plt.plot(daily_power_usage.index, daily_power_usage.values)
    plt.xlabel('Day')
    plt.ylabel('Total Daily Power Usage (k)')
    plt.title(f'Total Daily Power Usage for {location}, {equipment} - {year}/{month}')
    plt.show()

def plot_hourly_power(hourly_power_data):
    """
    Plot hourly power usage data for 7 days with hours from 0 to 24 in an interval of 4 hours.

    Parameters:
        hourly_power_data (DataFrame): DataFrame containing hourly power usage data.

    Returns:
        None
    """
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot hourly power usage for each day
    for day in range(7):
        # Filter data for the current day
        data_day = hourly_power_data[day * 24: (day + 1) * 24]

        # Plot the data
        sns.lineplot(x=data_day.index, y='Total Power Usage (k)', data=data_day, label=f'Day {day+1}', ax=ax)

    # Set plot labels and title
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Power Usage (k)')
    ax.set_title('Hourly Power Usage for 7 Days')
    ax.legend()
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_dataframe(dataframe):
    """
    Plot a DataFrame with figsize=(12, 8) and rotate the horizontal axis labels by 45 degrees.

    Parameters:
        dataframe (DataFrame): The DataFrame to be plotted.
    """
    ax = dataframe.plot(figsize=(24, 8))
    plt.xticks(rotation=45)
    plt.show()

def plot_acf_pacf(data, lag=48):
    """
    Plot the autocorrelation function (ACF) and partial autocorrelation function (PACF) for the data.

    Parameters:
        data (Series): Time series data.
        lag (int): Number of lags to include in the plots (default is 48).

    Returns:
        None
    """
    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot ACF
    sm.graphics.tsa.plot_acf(data.iloc[lag:], ax=ax1)
    ax1.set_title('Autocorrelation Function')

    # Plot PACF
    sm.graphics.tsa.plot_pacf(data.iloc[lag:], ax=ax2)
    ax2.set_title('Partial Autocorrelation Function')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_actual_vs_forecast(new_df2, start_index, end_index, dynamic=True):
    """
    Plot both the actual and forecasted power usage.

    Parameters:
        new_df2 (DataFrame): DataFrame containing power usage data and forecast.
        start_index (int): Start index for the forecast.
        end_index (int): End index for the forecast.
        dynamic (bool): Whether to use dynamic forecasting.

    Returns:
        None
    """
    # Generate forecast
    new_df2['forecast'] = model_fit.predict(start=start_index, end=end_index, dynamic=dynamic)

    # Plot actual vs forecasted power usage
    new_df2[['Used_Power', 'forecast']].plot(figsize=(12, 8))
    plt.xlabel('Timestamp')
    plt.ylabel('Power Usage')
    plt.title('Actual vs Forecasted Power Usage')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.show()


