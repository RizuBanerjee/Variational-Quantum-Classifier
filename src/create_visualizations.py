"""
create_visualizations.py

Creates final research figures from the
model comparison CSV.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "results/tables/"
    "final_model_comparison.csv"
)

OUTPUT_DIR = (
    "results/figures"
)


# ============================================================
# Load results
# ============================================================

def load_results():

    df = pd.read_csv(
        INPUT_FILE
    )

    return df


# ============================================================
# Classification Metrics
# ============================================================

def plot_classification_metrics(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    metrics = [
        "Test Accuracy (%)",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    # Convert accuracy from percentage
    # to the same 0-1 scale as other metrics.

    plot_df = df.copy()

    plot_df["Test Accuracy"] = (
        plot_df["Test Accuracy (%)"] / 100
    )

    x_labels = []

    for _, row in plot_df.iterrows():

        if row["Model"] == "Logistic Regression":

            x_labels.append(
                "Logistic Regression"
            )

        else:

            x_labels.append(
                f"VQC {row['Encoding']}"
            )

    values = {
        "Accuracy": plot_df["Test Accuracy"],
        "Precision": plot_df["Precision"],
        "Recall": plot_df["Recall"],
        "F1 Score": plot_df["F1 Score"]
    }

    x = range(
        len(x_labels)
    )

    width = 0.18

    plt.figure(
        figsize=(10, 6)
    )

    for index, metric in enumerate(
        values.keys()
    ):

        offset = (
            index - 1.5
        ) * width

        plt.bar(
            [
                position + offset
                for position in x
            ],
            values[metric],
            width=width,
            label=metric
        )

    plt.xticks(
        list(x),
        x_labels
    )

    plt.ylabel(
        "Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Classification Performance Comparison"
    )

    plt.ylim(
        0,
        1.1
    )

    plt.legend()

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/"
        "classification_metrics_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "Classification metrics graph saved to:"
    )

    print(
        output_file
    )


# ============================================================
# VQC Training Accuracy
# ============================================================

def plot_training_accuracy(df):

    vqc_df = df[
        df["Model"] == "VQC"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        vqc_df["Encoding"],
        vqc_df["Training Accuracy (%)"]
    )

    plt.xlabel(
        "Encoding"
    )

    plt.ylabel(
        "Training Accuracy (%)"
    )

    plt.title(
        "VQC Training Accuracy by Encoding"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/"
        "vqc_training_accuracy_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "\nVQC training accuracy graph saved to:"
    )

    print(
        output_file
    )


# ============================================================
# Best Training Loss
# ============================================================

def plot_best_loss(df):

    vqc_df = df[
        df["Model"] == "VQC"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        vqc_df["Encoding"],
        vqc_df["Best Loss"]
    )

    plt.xlabel(
        "Encoding"
    )

    plt.ylabel(
        "Best Training Loss"
    )

    plt.title(
        "Best Training Loss by VQC Encoding"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/"
        "vqc_best_loss_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "\nVQC best loss graph saved to:"
    )

    print(
        output_file
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CREATING FINAL RESEARCH FIGURES"
    )

    print(
        "=" * 70
    )

    df = load_results()

    plot_classification_metrics(
        df
    )

    plot_training_accuracy(
        df
    )

    plot_best_loss(
        df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VISUALIZATION CREATION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()