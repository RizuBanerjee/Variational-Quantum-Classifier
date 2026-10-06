"""
run_experiments.py

Complete evaluation pipeline for the VQC
feature encoding project.

Experiments:
    E1 - Full-range encoding
    E2 - Compressed encoding
    E3 - Reordered encoding

Also evaluates:
    Logistic Regression

Metrics:
    Accuracy
    Precision
    Recall
    F1 Score
    Training Loss
    Training Time
"""


import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_preprocessing import load_and_prepare_data
from src.quantum_model import (
    train_vqc,
    predict_dataset
)


# ============================================================
# Configuration
# ============================================================

ENCODINGS = [
    "E1",
    "E2",
    "E3"
]


# ============================================================
# Evaluate Quantum Model
# ============================================================

def evaluate_quantum_model(
    X_test,
    y_test,
    result
):
    """
    Calculate complete classification metrics
    for a trained binary VQC.
    """

    predictions = predict_dataset(
        X_test,
        result["weights"],
        result["encoding"]
    )

    # --------------------------------------------------------
    # Convert labels/predictions to standard integer arrays
    # --------------------------------------------------------

    y_true = [int(x) for x in y_test]
    y_pred = [int(x) for x in predictions]

    # --------------------------------------------------------
    # Verify binary labels
    # --------------------------------------------------------

    print("\nEvaluation label check:")
    print(
        "Actual labels    :",
        sorted(set(y_true))
    )
    print(
        "Predicted labels :",
        sorted(set(y_pred))
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm
    }

# ============================================================
# Run Quantum Experiments
# ============================================================

def run_quantum_experiments():

    results = []

    print("\n")
    print("=" * 70)
    print("QUANTUM ENCODING EXPERIMENTS")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_and_prepare_data()

    for encoding in ENCODINGS:

        print("\n")
        print("#" * 70)
        print(
            f"STARTING {encoding}"
        )
        print("#" * 70)

        result = train_vqc(
            encoding
        )

        metrics = evaluate_quantum_model(
            X_test,
            y_test,
            result
        )

        result["precision"] = metrics["precision"]
        result["recall"] = metrics["recall"]
        result["f1"] = metrics["f1"]
        result["confusion_matrix"] = (
            metrics["confusion_matrix"]
        )

        results.append(
            result
        )

    return results


# ============================================================
# Classical Baseline
# ============================================================

def run_classical_baseline():

    print("\n")
    print("=" * 70)
    print("CLASSICAL BASELINE - LOGISTIC REGRESSION")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_and_prepare_data()

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm
    }


# ============================================================
# Create Final Comparison Table
# ============================================================

def create_final_comparison(
    quantum_results,
    classical_result
):

    rows = []

    # --------------------------------------------------------
    # Classical model
    # --------------------------------------------------------

    rows.append({
        "Model": "Logistic Regression",
        "Encoding": "Classical",
        "Best Loss": None,
        "Best Loss Epoch": None,
        "Training Accuracy (%)": None,
        "Test Accuracy (%)":
            classical_result["accuracy"] * 100,
        "Precision":
            classical_result["precision"],
        "Recall":
            classical_result["recall"],
        "F1 Score":
            classical_result["f1"],
        "Training Time (s)": None
    })

    # --------------------------------------------------------
    # Quantum models
    # --------------------------------------------------------

    for result in quantum_results:

        rows.append({
            "Model": "VQC",
            "Encoding": result["encoding"],
            "Best Loss":
                result["best_loss"],
            "Best Loss Epoch":
                result["best_loss_epoch"],
            "Training Accuracy (%)":
                result["training_accuracy"] * 100,
            "Test Accuracy (%)":
                result["test_accuracy"] * 100,
            "Precision":
                result["precision"],
            "Recall":
                result["recall"],
            "F1 Score":
                result["f1"],
            "Training Time (s)":
                result["training_time"]
        })

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Round values
    # --------------------------------------------------------

    for column in [
        "Best Loss",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (s)"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(6)

    for column in [
        "Training Accuracy (%)",
        "Test Accuracy (%)"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(2)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    output_file = (
        "results/tables/"
        "final_model_comparison.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    return df


# ============================================================
# Confusion Matrices
# ============================================================

def save_confusion_matrices(
    quantum_results,
    classical_result
):

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Classical
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=
            classical_result["confusion_matrix"],
        display_labels=[
            "Setosa",
            "Versicolor"
        ]
    )

    display.plot()

    plt.title(
        "Logistic Regression Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "logistic_confusion_matrix_final.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Quantum
    # --------------------------------------------------------

    for result in quantum_results:

        display = ConfusionMatrixDisplay(
            confusion_matrix=
                result["confusion_matrix"],
            display_labels=[
                "Setosa",
                "Versicolor"
            ]
        )

        display.plot()

        plt.title(
            f"VQC {result['encoding']} "
            f"Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            "results/figures/"
            f"vqc_{result['encoding'].lower()}"
            "_confusion_matrix.png",
            dpi=300
        )

        plt.close()

    print(
        "\nConfusion matrices saved to:"
        "\nresults/figures/"
    )


# ============================================================
# Combined Loss Plot
# ============================================================

def create_loss_plot(
    quantum_results
):

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(9, 6)
    )

    for result in quantum_results:

        epochs = range(
            1,
            len(result["loss_history"]) + 1
        )

        plt.plot(
            epochs,
            result["loss_history"],
            marker="o",
            label=result["encoding"]
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        "VQC Training Loss Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_file = (
        "results/figures/"
        "final_vqc_loss_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nLoss comparison saved to:\n"
        f"{output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Quantum experiments
    # --------------------------------------------------------

    quantum_results = (
        run_quantum_experiments()
    )

    # --------------------------------------------------------
    # Classical baseline
    # --------------------------------------------------------

    classical_result = (
        run_classical_baseline()
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    create_final_comparison(
        quantum_results,
        classical_result
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    save_confusion_matrices(
        quantum_results,
        classical_result
    )

    # --------------------------------------------------------
    # Combined loss graph
    # --------------------------------------------------------

    create_loss_plot(
        quantum_results
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("COMPLETE MODEL EVALUATION FINISHED")
    print("=" * 70)


if __name__ == "__main__":

    main()