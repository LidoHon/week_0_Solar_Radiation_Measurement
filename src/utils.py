# utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Summary Statistics
def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, median, standard deviation, and other statistical measures."""
    return df.describe()

# Data Quality Check
def check_data_quality(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Check for missing values, negative values, and outliers in specified columns."""
    missing_values = df[columns].isnull().sum()
    outliers = df[columns].apply(lambda x: ((x < 0) | (x > x.quantile(0.99))).sum())
    return pd.DataFrame({'missing_values': missing_values, 'outliers': outliers})

# Time Series Analysis
def plot_time_series(df: pd.DataFrame, columns: list, title: str = "Time Series Analysis"):
    """Plot time series graphs for specified columns."""
    plt.figure(figsize=(14, 7))
    for column in columns:
        sns.lineplot(x=df.index, y=df[column], label=column)
    plt.title(title)
    plt.show()

# Correlation Analysis
def calculate_correlation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Calculate correlation matrix and plot a heatmap."""
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
    return corr_matrix

# Wind Analysis
def plot_polar_wind_analysis(df: pd.DataFrame, ws_col: str, wd_col: str):
    """Plot polar plot for wind speed and direction."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(np.radians(df[wd_col]), df[ws_col])
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    plt.title("Wind Speed and Direction")
    plt.show()

# Temperature Analysis
def analyze_temperature_influence(df: pd.DataFrame, temperature_cols: list, rh_col: str):
    """Examine how relative humidity might influence temperature readings."""
    for temp_col in temperature_cols:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[rh_col], y=df[temp_col])
        plt.title(f"RH vs {temp_col}")
        plt.show()

# Histograms
def plot_histograms(df: pd.DataFrame, columns: list):
    """Plot histograms for specified columns."""
    df[columns].hist(figsize=(14, 10), bins=20, edgecolor='black')
    plt.suptitle("Histograms")
    plt.show()

# Z-Score Analysis
def calculate_z_scores(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Calculate Z-scores for specified columns to detect outliers."""
    z_scores = df[columns].apply(zscore)
    return z_scores

# Bubble Charts
def plot_bubble_chart(df: pd.DataFrame, x_col: str, y_col: str, size_col: str, color_col: str):
    """Create a bubble chart to explore complex relationships."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], s=df[size_col] * 10, c=df[color_col], cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Bubble Chart: {x_col} vs {y_col}")
    plt.colorbar(label=color_col)
    plt.show()

# Data Cleaning
def clean_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Clean the dataset by handling anomalies and missing values."""
    df_cleaned = df.dropna(subset=columns)
    return df_cleaned
