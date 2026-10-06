"""
classical_baseline.py

Classical baseline for the VQC Feature Encoding project.

Model:
    Logistic Regression

Dataset:
    Binary Iris dataset

Features:
    Petal Length
    Petal Width

Classes:
    Setosa (0)
    Versicolor (1)
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

from data_preprocessing import load_and_prepare_data


def train_classical_model():
    """
    Train and evaluate Logistic Regression
    using the same data preparation pipeline
    that will later be used by the VQC.
    """

    # --------------------------------------------------
    # 1. Load prepared dataset
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_and_prepare_data()

    # --------------------------------------------------
    # 2. Create Logistic Regression model
    # --------------------------------------------------

    model = LogisticRegression(
        random_state=42
    )

    # --------------------------------------------------
    # 3. Train model
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------
    # 4. Make predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 5. Calculate evaluation metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------
    # 6. Print results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------
    # 7. Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # 8. Create result directories
    # --------------------------------------------------

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # --------------------------------------------------
    # 9. Plot confusion matrix
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Setosa", "Versicolor"]
    )

    display.plot()

    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "results/figures/logistic_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    results = pd.DataFrame({
        "Model": ["Logistic Regression"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1_Score": [f1]
    })

    results.to_csv(
        "results/tables/classical_baseline.csv",
        index=False
    )

    print(
        "\nResults saved to:"
        "\nresults/tables/classical_baseline.csv"
    )

    print(
        "Confusion matrix saved to:"
        "\nresults/figures/logistic_confusion_matrix.png"
    )

    return model, results


# ------------------------------------------------------
# Run the experiment when this file is executed directly
# ------------------------------------------------------

if __name__ == "__main__":

    model, results = train_classical_model()