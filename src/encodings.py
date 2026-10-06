"""
encodings.py

Feature encoding strategies for the
Variational Quantum Classifier project.

Input:
    Two normalized classical features in [0, 1]

Output:
    Two rotation angles in radians

Encoding variants:

    E1 - Full-range angle encoding
         [0, 1] -> [-pi, pi]

    E2 - Compressed angle encoding
         [0, 1] -> [-pi/2, pi/2]

    E3 - Reordered full-range encoding
         Same transformation as E1,
         but the two features are swapped.
"""

import numpy as np


# ============================================================
# Helper function
# ============================================================

def scale_to_range(x, new_min, new_max):
    """
    Map a normalized value from [0, 1]
    to [new_min, new_max].

    Formula:

        new_min + x * (new_max - new_min)

    Parameters
    ----------
    x : float
        Normalized feature value in [0, 1].

    new_min : float
        Lower bound of target angle range.

    new_max : float
        Upper bound of target angle range.

    Returns
    -------
    float
        Scaled rotation angle.
    """

    return new_min + x * (new_max - new_min)


# ============================================================
# E1 - Full-range encoding
# ============================================================

def encoding_e1(features):
    """
    E1: Full-range angle encoding.

    Maps:

        [0, 1] -> [-pi, pi]

    Feature 1 -> Qubit 0
    Feature 2 -> Qubit 1

    Parameters
    ----------
    features : array-like
        Two normalized features.

    Returns
    -------
    numpy.ndarray
        Two rotation angles.
    """

    x1 = features[0]
    x2 = features[1]

    angle_1 = scale_to_range(
        x1,
        -np.pi,
        np.pi
    )

    angle_2 = scale_to_range(
        x2,
        -np.pi,
        np.pi
    )

    return np.array([
        angle_1,
        angle_2
    ])


# ============================================================
# E2 - Compressed encoding
# ============================================================

def encoding_e2(features):
    """
    E2: Compressed angle encoding.

    Maps:

        [0, 1] -> [-pi/2, pi/2]

    Feature 1 -> Qubit 0
    Feature 2 -> Qubit 1

    Parameters
    ----------
    features : array-like
        Two normalized features.

    Returns
    -------
    numpy.ndarray
        Two rotation angles.
    """

    x1 = features[0]
    x2 = features[1]

    angle_1 = scale_to_range(
        x1,
        -np.pi / 2,
        np.pi / 2
    )

    angle_2 = scale_to_range(
        x2,
        -np.pi / 2,
        np.pi / 2
    )

    return np.array([
        angle_1,
        angle_2
    ])


# ============================================================
# E3 - Reordered encoding
# ============================================================

def encoding_e3(features):
    """
    E3: Reordered full-range encoding.

    Uses the same angle range as E1:

        [0, 1] -> [-pi, pi]

    but swaps the feature-to-qubit assignment.

    Feature 2 -> Qubit 0
    Feature 1 -> Qubit 1

    Parameters
    ----------
    features : array-like
        Two normalized features.

    Returns
    -------
    numpy.ndarray
        Two rotation angles.
    """

    x1 = features[0]
    x2 = features[1]

    # Feature 2 goes to qubit 0
    angle_for_qubit_0 = scale_to_range(
        x2,
        -np.pi,
        np.pi
    )

    # Feature 1 goes to qubit 1
    angle_for_qubit_1 = scale_to_range(
        x1,
        -np.pi,
        np.pi
    )

    return np.array([
        angle_for_qubit_0,
        angle_for_qubit_1
    ])


# ============================================================
# General encoding selector
# ============================================================

def get_encoding(name):
    """
    Return the encoding function corresponding
    to the requested encoding name.

    Parameters
    ----------
    name : str
        Encoding name:
            "E1"
            "E2"
            "E3"

    Returns
    -------
    function
        Corresponding encoding function.
    """

    encodings = {
        "E1": encoding_e1,
        "E2": encoding_e2,
        "E3": encoding_e3
    }

    if name not in encodings:
        raise ValueError(
            f"Unknown encoding '{name}'. "
            f"Choose from E1, E2, or E3."
        )

    return encodings[name]


# ============================================================
# Test the encodings
# ============================================================

if __name__ == "__main__":

    # Example normalized features
    test_features = np.array([
        0.25,
        0.75
    ])

    print("=" * 60)
    print("FEATURE ENCODING TEST")
    print("=" * 60)

    print("\nInput normalized features:")
    print(test_features)

    print("\nE1 - Full range [-pi, pi]:")
    print(encoding_e1(test_features))

    print("\nE2 - Compressed range [-pi/2, pi/2]:")
    print(encoding_e2(test_features))

    print("\nE3 - Reordered features:")
    print(encoding_e3(test_features))

    print("\n" + "=" * 60)
    print("ENCODING TEST COMPLETED")
    print("=" * 60)