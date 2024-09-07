# Exploratory-Data-Analysis
Conducted a comprehensive Exploratory Data Analysis on a real-time dataset spanning four locations and three different climate control equipment types

# **Hourly Power Usage Forecasting Using SARIMA**

This repository contains the code and methodology for predicting hourly power consumption using the **SARIMA** (Seasonal ARIMA) model. The project demonstrates how to model time series data with seasonal components to forecast future power usage.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## **Project Overview**

This project aims to predict future hourly power consumption using the **SARIMA** (Seasonal ARIMA) model. The project applies time series forecasting techniques to model and predict future power usage while accounting for seasonal patterns.

## **Features**

- Implements **SARIMA** for seasonal time series modelling.
- Automatic hyperparameter tuning using **Auto ARIMA**.
- Provides **visualization** of actual vs. predicted values.
- Forecasts for both short-term (next few hours) and long-term (up to 10 days).
- Includes support for **daily seasonality** (24-hour cycle).

## **Dataset**

- The project works with a dataset of **hourly power consumption** over several days.

### **Example Data Format:**

| timestamp           | power_usage |
|---------------------|-------------|
| 2024-09-01 00:00:00 | 1234        |
| 2024-09-01 01:00:00 | 1200        |
| ...                 | ...         |

- `power_usage`: The dependent variable representing hourly power consumption.

## **Methodology**

1. **Data Preprocessing**: 
   - Handling missing values and converting timestamps to datetime format.
   - Resampling data for hourly aggregation if necessary.

2. **Modeling**:
   - **SARIMA**: Seasonal ARIMA model captures time-based patterns and forecasts future values.

3. **Hyperparameter Tuning**:
   - **Auto ARIMA** is used to automate selecting the best parameters (`p`, `d`, `q`) for ARIMA and (`P`, `D`, `Q`, `m`) for seasonal ARIMA.

4. **Evaluation**:
   - Predictions are evaluated using metrics like **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.
   - Visual comparisons of predicted vs actual data are generated to assess the performance.
