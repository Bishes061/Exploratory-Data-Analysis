import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from holoviews import opts
import holoviews as hv
import plotly.express as px
from matplotlib.ticker import FuncFormatter
from matplotlib.markers import MarkerStyle
from dataProcessing_utils import merge_dataframes
from dataProcessing_utils import convert_numeric_month_to_name
from dataProcessing_utils import calculate_monthly_means
from dataProcessing_utils import calculate_monthly_totals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm


hv.extension('bokeh')


def plot_using_seaborn(data, x_col, y_col, x_lab, y_lab, title):
    """
    Plot a bar chart using Seaborn.

    Parameters:
        data (DataFrame): The DataFrame containing the data to be plotted.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        x_lab (str): Label for the x-axis.
        y_lab (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    sns.barplot(x=x_col, y=y_col, data=data)

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)

    plt.xticks(rotation=45)
    plt.show()


def create_hv_bar_chart(data, x_col, y_col, xlabel, ylabel, title):
    """
    Create a horizontal bar chart using HoloViews.

    Parameters:
    - data : DataFrame
        The input DataFrame containing the data.
    - x_col : str
        The column name for the x-axis.
    - y_col : str
        The column name for the y-axis.
    - xlabel : str
        Label for the x-axis.
    - ylabel : str
        Label for the y-axis.
    - title : str
        Title for the plot.

    Returns:
    - bar_chart : hv.Bars
        A horizontal bar chart representing the data.
    """
    bar_chart = hv.Bars(data, [x_col], [y_col])

    bar_chart = bar_chart.opts(
        opts.Bars(
            width=800, height=400, tools=['hover'],
            xlabel=xlabel, ylabel=ylabel, title=title,
            color='blue', show_grid = True, invert_axes=False
        )
    )

    return bar_chart

def plot_ploty(data, x_col, y_col, title, labels_dict, color_col, barmode='group', xaxis_tickangle=-45, size = (800, 600)):
    """
    Generate a Plotly bar chart.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        title (str): The title for the plot.
        labels_dict (dict): A dictionary specifying the labels for x and y axes.
        color_col (str): The column name to map colors to.
        barmode (str, optional): The mode of the bars ('group', 'overlay', 'relative'). Defaults to 'group'.
        xaxis_tickangle (int, optional): The angle of rotation for x-axis tick labels. Defaults to -45.
        size (tuple, optional): The size of the plot (width, height) in pixels. Defaults to (800, 600).

    Returns:
        None: The function displays the plot using Plotly's interactive interface.
    """
    fig = px.bar(data, x=x_col, y=y_col, title=title, labels=labels_dict,
                 color=color_col, barmode=barmode)

    fig.update_layout(xaxis_tickangle=xaxis_tickangle, width=size[0], height=size[1])

    fig.show()


def plot_dual_axis_timeseries(df, x_column, y1_column, y2_column, y1_label, y2_label, y1_color, y2_color, y1_lim, y2_lim, y1_formatter, title, xlabel, figsize=(24, 8)):
    """
    Plot two sets of data on a shared x-axis with dual y-axes.

    Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        x_column (str): Column name for x-axis data.
        y1_column (str): Column name for primary y-axis data.
        y2_column (str): Column name for secondary y-axis data.
        y1_label (str): Label for the primary y-axis.
        y2_label (str): Label for the secondary y-axis.
        y1_color (str): Color for the primary y-axis data.
        y2_color (str): Color for the secondary y-axis data.
        y1_lim (tuple): Limits for the primary y-axis (min, max).
        y2_lim (tuple): Limits for the secondary y-axis (min, max).
        y1_formatter (FuncFormatter): Formatter for primary y-axis ticks.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        figsize (tuple): Size of the figure (width, height).
    """
    # Create subplots with shared x-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot bar chart for primary y-axis data
    sns.barplot(x=x_column, y=y1_column, data=df, ax=ax1, color=y1_color, label=y1_label, width=0.5)
    ax1.set_ylabel(y1_label, color=y1_color)

    # Create a second Y-axis for the secondary y-axis data
    ax2 = ax1.twinx()
    sns.lineplot(x=x_column, y=y2_column, data=df, ax=ax2, color=y2_color, marker='D', label=y2_label, linestyle='dotted')
    ax2.set_ylabel(y2_label, color=y2_color)

    # Set Y-axis limits
    ax1.set_ylim(y1_lim)
    ax2.set_ylim(y2_lim)

    # Customize y-axis ticks for primary y-axis data
    ax1.yaxis.set_major_formatter(y1_formatter)

    # Set x-axis label
    ax1.set_xlabel(xlabel)

    # Set x-axis tick labels rotation
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Add legends by combining from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Add title
    plt.title(title)

    # Show the plot
    plt.show()

def y_axis_formatter(x, pos):
    """
    Format y-axis ticks to display values in thousands with 'k' suffix.

    Parameters:
        x (float): The tick value on the y-axis.
        pos (int): The position of the tick.

    Returns:
        str: The formatted tick label.
    """
    if x != 0:
        return '{:,.0f}k'.format(x / 1000)
    else:
        return '0'


def generate_scatter_plot(merged_df, x, y, hue, palette, legend, y_axis_formatter):
    """
    Plot a scatter plot between Total Power Usage and Mean Temperature for all the 12 months.

    Parameters:
        merged_df (DataFrame): Merged DataFrame containing power consumption and temperature data.
        x (str): The column name for the x-axis (default is 'temperature').
        y (str): The column name for the y-axis (default is 'used_power').
        hue (str): The column name whose values will be used to color the points (default is 'month').
        palette (str or list of colors): Set of colors for mapping the hue variable (default is 'viridis').
        legend (str): How to draw the legend (default is 'full').
        y_axis_formatter (function): A function to format the y-axis ticks.
    """
    # Create a scatter plot
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=x, y=y, data=merged_df, hue=hue, palette=palette, legend=legend)

    # Adjust y-axis ticks using the custom formatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    # Adding title and labels
    plt.title('Scatter Plot: Mean Temperature vs. Total Power Usage')
    plt.xlabel('Mean Temperature')
    plt.ylabel('Total Power Usage (in 100k)')

    # Set Y-axis and X-axis limits
    plt.ylim(0, 3e6)
    plt.xlim(0, 35)

    # Show the plot
    plt.show()


def plot_line_chart_seaborn(df, x_col, y_col, xlabel, ylabel, title):
    """
    Generate a line chart using Seaborn.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(24, 8))
    sns.lineplot(data=df, x=x_col, y=y_col, marker='D', linestyle='dotted', color='red', label='Temperature')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_area_chart(df, x_col, y_col, xlabel, ylabel, title):
    """
    Generate an area chart using Seaborn and Matplotlib.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(24, 8))
    sns.lineplot(x=x_col, y=y_col, data=df, color='skyblue', linewidth=0, alpha=0.6)
    plt.fill_between(df[x_col], df[y_col], color='skyblue', alpha=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_dual_axis_chart(df, x_col, y1_col, y2_col, xlabel, y1_label, y2_label, title):
    """
    Plot a combined line chart with dual y-axes for two different columns in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y1_col (str): The column name for the first y-axis.
        y2_col (str): The column name for the second y-axis.
        xlabel (str): The label for the x-axis.
        y1_label (str): The label for the first y-axis.
        y2_label (str): The label for the second y-axis.
        title (str): The title of the plot.
    """
    # Create a figure and an axis object
    fig, ax1 = plt.subplots(figsize=(24, 8))

    # Plot the first line chart with area plot for power consumption
    sns.lineplot(data=df, x=x_col, y=y1_col, ax=ax1, color='darkblue', linewidth=1, alpha=0.6, label=y1_label)
    ax1.fill_between(df[x_col], df[y1_col], color='darkblue', alpha=0.4)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for the mean temperature line plot
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=x_col, y=y2_col, ax=ax2, linestyle='-', color='red', label=y2_label)
    ax2.set_ylabel(y2_label, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set Y-axis limits
    ax1.set_ylim([0, 200000])
    ax2.set_ylim([0, 30])

    # Customize y-axis ticks for power consumption
    formatter = FuncFormatter(lambda x, _: '{:,.0f}k'.format(x/1000))  # Convert to k units
    ax1.yaxis.set_major_formatter(formatter)

    # Add Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Set title and labels
    plt.title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)

    # Show the plot
    fig.tight_layout()  # Adjusts plot to fit titles
    plt.show()


#MODELS
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

def plot_acf_pacf(series, lags=None):
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

