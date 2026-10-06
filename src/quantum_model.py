"""
quantum_model.py

Variational Quantum Classifier with selectable
feature encoding.

Supported encodings:
    E1 - Full range [-pi, pi]
    E2 - Compressed range [-pi/2, pi/2]
    E3 - Reordered features using [-pi, pi]

The VQC architecture, optimizer, learning rate,
number of layers, epochs, initialization and
dataset are kept identical for all encodings.

This allows a controlled ablation study.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

from src.data_preprocessing import load_and_prepare_data
from src.encodings import get_encoding


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 2
N_LAYERS = 2

LEARNING_RATE = 0.1
EPOCHS = 30

SEED = 42


# ============================================================
# Quantum device
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# Feature encoding
# ============================================================

def encode_features(features, encoding_name):
    """
    Apply the selected feature encoding.

    Parameters
    ----------
    features : array-like
        Two normalized features.

    encoding_name : str
        E1, E2 or E3.

    Returns
    -------
    None
    """

    encoding_function = get_encoding(
        encoding_name
    )

    angles = encoding_function(
        features
    )

    qml.RX(
        angles[0],
        wires=0
    )

    qml.RX(
        angles[1],
        wires=1
    )


# ============================================================
# VQC Circuit
# ============================================================

@qml.qnode(dev)
def quantum_circuit(
    features,
    weights,
    encoding_name
):
    """
    Common VQC circuit used by E1, E2 and E3.
    """

    # --------------------------------------------------------
    # Feature encoding
    # --------------------------------------------------------

    encode_features(
        features,
        encoding_name
    )

    # --------------------------------------------------------
    # Trainable layers
    # --------------------------------------------------------

    for layer in range(N_LAYERS):

        # Qubit 0
        qml.RY(
            weights[layer, 0, 0],
            wires=0
        )

        qml.RZ(
            weights[layer, 0, 1],
            wires=0
        )

        # Qubit 1
        qml.RY(
            weights[layer, 1, 0],
            wires=1
        )

        qml.RZ(
            weights[layer, 1, 1],
            wires=1
        )

        # Entanglement
        qml.CNOT(
            wires=[0, 1]
        )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# Prediction
# ============================================================

def predict_single(
    features,
    weights,
    encoding_name
):
    """
    Predict one sample.

    Expectation value >= 0 -> class 1
    Expectation value <  0 -> class 0
    """

    output = quantum_circuit(
        features,
        weights,
        encoding_name
    )

    if output >= 0:
        return 1

    return 0


def predict_dataset(
    X,
    weights,
    encoding_name
):
    """
    Predict all samples in X.
    """

    predictions = []

    for features in X:

        prediction = predict_single(
            features,
            weights,
            encoding_name
        )

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    )


# ============================================================
# Loss function
# ============================================================

def loss_function(
    X,
    y,
    weights,
    encoding_name
):
    """
    Mean squared error loss.

    Class 0 -> -1
    Class 1 -> +1
    """

    total_loss = 0.0

    for features, label in zip(X, y):

        output = quantum_circuit(
            features,
            weights,
            encoding_name
        )

        target = 2 * label - 1

        total_loss += (
            output - target
        ) ** 2

    return total_loss / len(X)


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    X,
    y,
    weights,
    encoding_name
):
    """
    Calculate classification accuracy.
    """

    predictions = predict_dataset(
        X,
        weights,
        encoding_name
    )

    correct = np.sum(
        predictions == y
    )

    return correct / len(y)


# ============================================================
# Train selected encoding
# ============================================================

def train_vqc(encoding_name):
    """
    Train the VQC using the selected encoding.

    Parameters
    ----------
    encoding_name : str
        E1, E2 or E3.

    Returns
    -------
    dict
        Experiment results.
    """

    valid_encodings = [
        "E1",
        "E2",
        "E3"
    ]

    if encoding_name not in valid_encodings:

        raise ValueError(
            f"Invalid encoding: {encoding_name}. "
            f"Choose E1, E2 or E3."
        )

    print("\n" + "=" * 65)
    print(
        f"VQC EXPERIMENT - ENCODING {encoding_name}"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_and_prepare_data()

    # --------------------------------------------------------
    # Encoding information
    # --------------------------------------------------------

    encoding_ranges = {
        "E1": "[-pi, pi]",
        "E2": "[-pi/2, pi/2]",
        "E3": "[-pi, pi] + feature reorder"
    }

    print("\n" + "=" * 65)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 65)

    print(
        f"Encoding        : {encoding_name}"
    )

    print(
        f"Encoding scheme : "
        f"{encoding_ranges[encoding_name]}"
    )

    print(
        f"Qubits          : {N_QUBITS}"
    )

    print(
        f"VQC layers      : {N_LAYERS}"
    )

    print(
        "Optimizer       : Adam"
    )

    print(
        f"Learning rate   : {LEARNING_RATE}"
    )

    print(
        f"Epochs          : {EPOCHS}"
    )

    print(
        f"Random seed     : {SEED}"
    )

    # --------------------------------------------------------
    # Initialize weights
    # --------------------------------------------------------

    np.random.seed(SEED)

    weights = np.array(
        0.01 * np.random.randn(
            N_LAYERS,
            N_QUBITS,
            2
        ),
        requires_grad=True
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = qml.AdamOptimizer(
        stepsize=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    loss_history = []
    accuracy_history = []

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        f"TRAINING {encoding_name}"
    )
    print("=" * 65)

    for epoch in range(EPOCHS):

        weights, current_loss = (
            optimizer.step_and_cost(
                lambda w: loss_function(
                    X_train,
                    y_train,
                    w,
                    encoding_name
                ),
                weights
            )
        )

        train_accuracy = calculate_accuracy(
            X_train,
            y_train,
            weights,
            encoding_name
        )

        loss_history.append(
            float(current_loss)
        )

        accuracy_history.append(
            float(train_accuracy)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {current_loss:.6f} | "
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Training time
    # --------------------------------------------------------

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Test accuracy
    # --------------------------------------------------------

    test_accuracy = calculate_accuracy(
        X_test,
        y_test,
        weights,
        encoding_name
    )

    # --------------------------------------------------------
    # Best loss and convergence
    # --------------------------------------------------------

    best_loss = min(
        loss_history
    )

    best_loss_epoch = (
        loss_history.index(best_loss) + 1
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        f"{encoding_name} FINAL RESULTS"
    )
    print("=" * 65)

    print(
        f"Final training loss : "
        f"{loss_history[-1]:.6f}"
    )

    print(
        f"Best training loss  : "
        f"{best_loss:.6f}"
    )

    print(
        f"Best loss epoch     : "
        f"{best_loss_epoch}"
    )

    print(
        f"Training accuracy   : "
        f"{accuracy_history[-1] * 100:.2f}%"
    )

    print(
        f"Test accuracy       : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Training time       : "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Create result directories
    # --------------------------------------------------------

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    # ========================================================
    # Save loss curve
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(1, EPOCHS + 1),
        loss_history,
        marker="o"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        f"VQC Training Loss - Encoding {encoding_name}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/figures/"
        f"vqc_{encoding_name.lower()}_loss_curve.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Save training history
    # ========================================================

    history_file = (
        "results/tables/"
        f"vqc_{encoding_name.lower()}_training_history.csv"
    )

    with open(
        history_file,
        "w"
    ) as file:

        file.write(
            "epoch,loss,train_accuracy\n"
        )

        for i in range(EPOCHS):

            file.write(
                f"{i + 1},"
                f"{loss_history[i]},"
                f"{accuracy_history[i]}\n"
            )

    # ========================================================
    # Save summary
    # ========================================================

    summary_file = (
        "results/tables/"
        f"vqc_{encoding_name.lower()}_result.csv"
    )

    with open(
        summary_file,
        "w"
    ) as file:

        file.write(
            "Encoding,"
            "Angle_Range,"
            "Final_Loss,"
            "Best_Loss,"
            "Best_Loss_Epoch,"
            "Training_Accuracy,"
            "Test_Accuracy,"
            "Training_Time\n"
        )

        file.write(
            f"{encoding_name},"
            f"{encoding_ranges[encoding_name]},"
            f"{loss_history[-1]},"
            f"{best_loss},"
            f"{best_loss_epoch},"
            f"{accuracy_history[-1]},"
            f"{test_accuracy},"
            f"{training_time}\n"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print(
        "\nLoss curve saved to:"
    )

    print(
        f"results/figures/"
        f"vqc_{encoding_name.lower()}_loss_curve.png"
    )

    print(
        "\nTraining history saved to:"
    )

    print(
        f"results/tables/"
        f"vqc_{encoding_name.lower()}_training_history.csv"
    )

    print(
        "\nSummary saved to:"
    )

    print(
        f"results/tables/"
        f"vqc_{encoding_name.lower()}_result.csv"
    )

    return {
        "encoding": encoding_name,
        "final_loss": float(loss_history[-1]),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_loss_epoch),
        "training_accuracy": float(accuracy_history[-1]),
        "test_accuracy": float(test_accuracy),
        "training_time": float(training_time),
        "weights": weights,
        "loss_history": [
            float(loss)
            for loss in loss_history
        ],
        "accuracy_history": [
            float(acc)
            for acc in accuracy_history
        ]
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Default encoding
    # --------------------------------------------------------

    encoding = "E1"

    # --------------------------------------------------------
    # Allow command-line argument
    #
    # Example:
    #
    # python -m src.quantum_model E2
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        encoding = sys.argv[1].upper()

    train_vqc(
        encoding
    )