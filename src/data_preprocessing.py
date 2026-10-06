"""
data_preprocessing.py

Prepares the Iris dataset for our VQC project.

Dataset:
    Iris

Selected classes:
    Setosa (0)
    Versicolor (1)

Selected features:
    Petal Length
    Petal Width

Processing:
    1. Load Iris dataset
    2. Select two classes
    3. Select two features
    4. Train/test split
    5. Min-Max scaling
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data():
    """
    Load and prepare the binary Iris dataset.

    Returns
    -------
    X_train_scaled : numpy.ndarray
        Scaled training features.

    X_test_scaled : numpy.ndarray
        Scaled testing features.

    y_train : numpy.ndarray
        Training labels.

    y_test : numpy.ndarray
        Testing labels.

    scaler : MinMaxScaler
        Fitted scaler.
    """

    # --------------------------------------------------
    # 1. Load Iris dataset
    # --------------------------------------------------

    iris = load_iris()

    X = iris.data
    y = iris.target

    print("=" * 60)
    print("ORIGINAL IRIS DATASET")
    print("=" * 60)

    print(f"Number of samples : {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes           : {iris.target_names}")

    # --------------------------------------------------
    # 2. Select two classes
    #
    # 0 = Setosa
    # 1 = Versicolor
    # 2 = Virginica
    # --------------------------------------------------

    class_mask = (y == 0) | (y == 1)

    X_binary = X[class_mask]
    y_binary = y[class_mask]

    # --------------------------------------------------
    # 3. Select two features
    #
    # Feature index:
    # 0 -> Sepal Length
    # 1 -> Sepal Width
    # 2 -> Petal Length
    # 3 -> Petal Width
    # --------------------------------------------------

    X_selected = X_binary[:, [2, 3]]

    feature_names = [
        "petal_length",
        "petal_width"
    ]

    df = pd.DataFrame(
        X_selected,
        columns=feature_names
    )

    df["target"] = y_binary

    print("\n" + "=" * 60)
    print("SELECTED DATASET")
    print("=" * 60)

    print(f"Number of samples : {X_selected.shape[0]}")
    print(f"Number of features: {X_selected.shape[1]}")
    print("Classes           : Setosa (0), Versicolor (1)")

    print("\nClass distribution:")
    print(df["target"].value_counts().sort_index())

    # --------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_binary,
        test_size=0.20,
        random_state=42,
        stratify=y_binary
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining class distribution:")
    print(np.bincount(y_train))

    print("\nTesting class distribution:")
    print(np.bincount(y_test))

    # --------------------------------------------------
    # 5. Scale features
    #
    # IMPORTANT:
    # Fit the scaler ONLY on training data.
    # --------------------------------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    print("Scaling method: Min-Max Scaling")

    print("\nFirst 5 training samples after scaling:")
    print(X_train_scaled[:5])

    print("\nTraining feature minimum:")
    print(X_train_scaled.min(axis=0))

    print("\nTraining feature maximum:")
    print(X_train_scaled.max(axis=0))

    print("\nData preparation completed successfully!")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ------------------------------------------------------
# Run this file directly
# ------------------------------------------------------

if __name__ == "__main__":

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = load_and_prepare_data()