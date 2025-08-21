"""
Utility functions for Credit Card Default Detection project.

This module contains helper functions for data preprocessing, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Dict, Any, List
import config

warnings.filterwarnings("ignore")


def create_directories() -> None:
    """Create necessary directories for the project."""
    import os

    directories = [config.PLOTS_DIR, config.MODELS_DIR, config.RESULTS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing purposes if actual data is not available.

    Returns:
        Tuple of (train_data, test_data) DataFrames
    """
    np.random.seed(config.RANDOM_STATE)

    # Generate sample data
    n_train = 25000
    n_test = 5000
    n_features = 23

    # Create feature names
    feature_names = [f"X{i}" for i in range(1, n_features + 1)]

    # Generate training data
    train_data = pd.DataFrame(
        np.random.randn(n_train, n_features), columns=feature_names
    )

    # Add ID column
    train_data.insert(0, "id", range(1, n_train + 1))

    # Generate target variable (imbalanced)
    train_data["Y"] = np.random.choice([0, 1], size=n_train, p=[0.8, 0.2])

    # Generate test data
    test_data = pd.DataFrame(np.random.randn(n_test, n_features), columns=feature_names)

    # Add ID column
    test_data.insert(0, "id", range(1, n_test + 1))

    return train_data, test_data


def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and outliers.

    Args:
        df: Input DataFrame
        is_training: Whether this is training data

    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()

    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            if is_training:
                # For training data, fill with median
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
            else:
                # For test data, use 0 as default
                df_processed[col].fillna(0, inplace=True)

    # Handle outliers using IQR method for continuous variables
    if is_training:
        continuous_cols = [
            col
            for col in df_processed.columns
            if col not in ["id", "Y"]
            and len(df_processed[col].unique()) > config.CATEGORICAL_THRESHOLD
        ]

        for col in continuous_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers instead of removing them
            df_processed[col] = df_processed[col].clip(
                lower=lower_bound, upper=upper_bound
            )

    return df_processed


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comprehensive feature summary DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Feature summary DataFrame
    """
    summary_data = []

    for col in df.columns:
        col_info = {
            "Feature": col,
            "Description": config.FEATURE_DESCRIPTIONS.get(col, "Unknown"),
            "Data Type": str(df[col].dtype),
            "Missing Values": df[col].isnull().sum(),
            "Missing %": (df[col].isnull().sum() / len(df)) * 100,
            "Unique Values": df[col].nunique(),
            "Min": df[col].min() if df[col].dtype in ["int64", "float64"] else None,
            "Max": df[col].max() if df[col].dtype in ["int64", "float64"] else None,
            "Mean": df[col].mean() if df[col].dtype in ["int64", "float64"] else None,
            "Std": df[col].std() if df[col].dtype in ["int64", "float64"] else None,
        }
        summary_data.append(col_info)

    return pd.DataFrame(summary_data)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_plot: bool = True,
    output_dir: str = "plots",
) -> None:
    """
    Plot confusion matrix with detailed annotations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_plot: Whether to save the plot
        output_dir: Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )

    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    plt.tight_layout()

    if save_plot:
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            f"{output_dir}/confusion_matrix.png",
            dpi=config.PLOT_DPI,
            bbox_inches="tight",
        )

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_plot: bool = True,
    output_dir: str = "plots",
) -> None:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_plot: Whether to save the plot
        output_dir: Directory to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            f"{output_dir}/roc_curve.png", dpi=config.PLOT_DPI, bbox_inches="tight"
        )

    plt.show()


def plot_feature_importance(
    model, feature_names: List[str], save_plot: bool = True, output_dir: str = "plots"
) -> None:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_plot: Whether to save the plot
        output_dir: Directory to save the plot
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_ attribute")
        return

    # Get feature importance
    importances = model.feature_importances_

    # Create DataFrame for sorting
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Plot top 20 features
    top_features = feature_importance_df.head(20)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title("Top 20 Feature Importances", fontsize=16, pad=20)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
        )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_plot:
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            f"{output_dir}/feature_importance.png",
            dpi=config.PLOT_DPI,
            bbox_inches="tight",
        )

    plt.show()


def save_results_to_file(
    results: Dict[str, Any], filename: str, output_dir: str = "results"
) -> None:
    """
    Save results to a text file.

    Args:
        results: Dictionary containing results
        filename: Name of the file to save
        output_dir: Directory to save the file
    """
    import os
    import json
    from datetime import datetime

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)

    # Add timestamp
    results["timestamp"] = datetime.now().isoformat()

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4, default=str)

    print(f"Results saved to: {filepath}")


def print_model_comparison(models_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison of model performances.

    Args:
        models_metrics: Dictionary containing metrics for each model
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models_metrics).T

    # Round metrics to 4 decimal places
    comparison_df = comparison_df.round(4)

    # Print the comparison
    print(comparison_df.to_string())

    # Find best model for each metric
    print("\n" + "-" * 80)
    print("BEST PERFORMING MODELS BY METRIC")
    print("-" * 80)

    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"{metric:15}: {best_model:20} ({best_score:.4f})")


def create_model_card(
    model_name: str,
    model_performance: Dict[str, float],
    model_params: Dict[str, Any],
    output_dir: str = "results",
) -> None:
    """
    Create a model card with detailed information about the model.

    Args:
        model_name: Name of the model
        model_performance: Performance metrics
        model_params: Model parameters
        output_dir: Directory to save the model card
    """
    import os
    from datetime import datetime

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, f"{model_name}_model_card.md")

    with open(filepath, "w") as f:
        f.write(f"# Model Card: {model_name}\n\n")
        f.write(f"**Date Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in model_performance.items():
            f.write(f"| {metric} | {value:.4f} |\n")

        f.write("\n## Model Parameters\n\n")
        f.write("```json\n")
        import json

        f.write(json.dumps(model_params, indent=2))
        f.write("\n```\n")

        f.write("\n## Model Description\n\n")
        f.write("This model was trained on credit card default detection data.\n")
        f.write(
            "The target variable indicates whether a customer defaulted (1) or not (0).\n"
        )

        f.write("\n## Data Description\n\n")
        f.write("The model uses the following features:\n")
        for feature, description in config.FEATURE_DESCRIPTIONS.items():
            if feature not in ["id", "Y"]:
                f.write(f"- **{feature}**: {description}\n")

    print(f"Model card saved to: {filepath}")
