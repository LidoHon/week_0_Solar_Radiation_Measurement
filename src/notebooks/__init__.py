import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

# Load the datasets
benin_df = pd.read_csv('data/benin-malanville.csv')
sierraleone_df = pd.read_csv('data/sierraleone-bumbuna.csv')
togo_df = pd.read_csv('data/togo-dapaong_qc.csv')

# Display the first few rows of each DataFrame
# print("Benin Data:")
# print(benin_df.head())

# print("\nSierra Leone Data:")
# print(sierraleone_df.head())

# print("\nTogo Data:")
# print(togo_df.head())

# Summary Statistics
def print_summary_stats(df, name):
    print(f"Summary statistics for {name}:")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Print summary statistics
    print(numeric_df.describe())

# Example usage of the function
# print_summary_stats(benin_df, "Benin")
# print_summary_stats(sierraleone_df, "Sierra Leone")
# print_summary_stats(togo_df, "Togo")



# Create a function to plot histograms and save them
def plot_histograms(df, name):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.hist(figsize=(12, 10), bins=30)
    plt.suptitle(f"Histograms for {name}")
    plt.savefig(f'{name}_histograms.png')
    plt.close()

# Create a function to plot scatter plots and save them
def plot_scatter_plots(df, x_col, y_col, name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f"Scatter Plot of {x_col} vs {y_col} for {name}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(f'{name}_{x_col}_vs_{y_col}.png')
    plt.close()

# Create a function to plot a correlation heatmap and save it
def plot_correlation_heatmap(df, name):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Heatmap for {name}")
    plt.savefig(f'{name}_correlation_heatmap.png')
    plt.close()

# Example usage:
plot_histograms(benin_df, "Benin")
plot_scatter_plots(benin_df, 'GHI', 'DNI', "Benin")
plot_correlation_heatmap(benin_df, "Benin")

