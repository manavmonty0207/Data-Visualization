"""
Experiment 1: Data Preprocessing and Visualization
Beginner-friendly version.

Run:
    python main.py

This program processes three CSV datasets, saves cleaned data and creates plots.
The housing dataset is an illustrative sample; diabetes and wine are built-in
scikit-learn datasets (when generated using the provided setup).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder locations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def process_dataset(file_name):
    """Clean one dataset, scale numeric columns and save visualizations."""

    print("\n" + "=" * 60)
    print("PROCESSING:", file_name)
    print("=" * 60)

    # STEP 1: Read the CSV file
    df = pd.read_csv(DATA_DIR / file_name)
    print("\nOriginal dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    # STEP 2: Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    print("\nNumeric columns:", numeric_columns)
    print("Categorical columns:", categorical_columns)

    # STEP 3: Handle missing values
    missing_before = df.isnull().sum().sum()
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])

    print("\nMissing values before cleaning:", missing_before)
    print("Missing values after cleaning:", df.isnull().sum().sum())

    # STEP 4: Remove duplicate rows
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    print("\nDuplicate rows removed:", duplicates_before)
    print("Shape after removing duplicates:", df.shape)

    # STEP 5: Handle outliers using the IQR method (numeric columns only)
    # IQR = Q3 - Q1. Values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
    # are treated as potential outliers. We cap them at these limits.
    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)

    print("\nOutlier handling completed using IQR capping.")

    # Save cleaned dataset
    dataset_name = Path(file_name).stem
    result_folder = OUTPUT_DIR / dataset_name
    result_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_folder / "cleaned_data.csv", index=False)

    # STEP 6: Select numeric columns for scaling
    numeric_data = df[numeric_columns].copy()

    # Min-Max Normalization: (x - minimum) / (maximum - minimum)
    minimum = numeric_data.min()
    maximum = numeric_data.max()
    minmax_data = (numeric_data - minimum) / (maximum - minimum).replace(0, 1)

    # Z-score Standardization: (x - mean) / standard deviation
    mean = numeric_data.mean()
    std = numeric_data.std(ddof=0).replace(0, 1)
    zscore_data = (numeric_data - mean) / std

    minmax_data.to_csv(result_folder / "minmax_normalized.csv", index=False)
    zscore_data.to_csv(result_folder / "zscore_standardized.csv", index=False)

    print("\nMin-Max sample (first 5 rows):")
    print(minmax_data.head().round(3))
    print("\nZ-score sample (first 5 rows):")
    print(zscore_data.head().round(3))

    # STEP 7: Bar plot - mean of each numeric feature
    plt.figure(figsize=(10, 5))
    numeric_data.mean().plot(kind="bar")
    plt.title("Mean of Numeric Features - " + dataset_name)
    plt.xlabel("Features")
    plt.ylabel("Mean value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(result_folder / "bar_plot.png", dpi=150)
    plt.close()

    # STEP 8: Box plots before and after scaling
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    numeric_data.boxplot(ax=axes[0], rot=90)
    axes[0].set_title("Before scaling")
    minmax_data.boxplot(ax=axes[1], rot=90)
    axes[1].set_title("After Min-Max")
    zscore_data.boxplot(ax=axes[2], rot=90)
    axes[2].set_title("After Z-score")
    fig.suptitle("Box Plot Comparison - " + dataset_name)
    plt.tight_layout()
    plt.savefig(result_folder / "box_plot_comparison.png", dpi=150)
    plt.close()

    # STEP 9: Histograms before and after scaling (first 3 numeric columns)
    selected_columns = numeric_columns[:3]
    fig, axes = plt.subplots(len(selected_columns), 3, figsize=(12, 4 * len(selected_columns)))
    if len(selected_columns) == 1:
        axes = np.array([axes])
    for row, column in enumerate(selected_columns):
        axes[row, 0].hist(numeric_data[column], bins=15, edgecolor="black")
        axes[row, 0].set_title(column + " - Original")
        axes[row, 1].hist(minmax_data[column], bins=15, edgecolor="black")
        axes[row, 1].set_title(column + " - Min-Max")
        axes[row, 2].hist(zscore_data[column], bins=15, edgecolor="black")
        axes[row, 2].set_title(column + " - Z-score")
    fig.suptitle("Histogram Comparison - " + dataset_name)
    plt.tight_layout()
    plt.savefig(result_folder / "histogram_comparison.png", dpi=150)
    plt.close()

    # STEP 10: Scatter plot before and after scaling (first two numeric columns)
    if len(numeric_columns) >= 2:
        x_col = numeric_columns[0]
        y_col = numeric_columns[1]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].scatter(numeric_data[x_col], numeric_data[y_col], alpha=0.7)
        axes[0].set_title("Original")
        axes[1].scatter(minmax_data[x_col], minmax_data[y_col], alpha=0.7)
        axes[1].set_title("Min-Max")
        axes[2].scatter(zscore_data[x_col], zscore_data[y_col], alpha=0.7)
        axes[2].set_title("Z-score")
        for ax in axes:
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        fig.suptitle("Scatter Plot Comparison - " + dataset_name)
        plt.tight_layout()
        plt.savefig(result_folder / "scatter_comparison.png", dpi=150)
        plt.close()

    print("\nSaved cleaned data and plots in:", result_folder)
    print("DONE:", file_name)


if __name__ == "__main__":
    # Run the same experiment on all three datasets
    datasets = ["housing_sample.csv", "diabetes.csv", "wine.csv"]
    for dataset in datasets:
        process_dataset(dataset)

    print("\nAll datasets processed successfully!")
