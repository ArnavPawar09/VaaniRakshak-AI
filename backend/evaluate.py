from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "vaani_classifier.joblib"
TEST_FEATURES_PATH = PROJECT_ROOT / "features" / "X_test.npy"
TEST_LABELS_PATH = PROJECT_ROOT / "features" / "y_test.npy"

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "features" / "confusion_matrix.png"
)


def evaluate():

    print("Loading VaaniRakshak classifier...")

    classifier = joblib.load(MODEL_PATH)

    X_test = np.load(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    print("Model and test data loaded.")
    print(f"Test samples: {len(y_test)}")

    # Predictions
    predictions = classifier.predict(X_test)

    probabilities = classifier.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print()
    print("========== VAANIRAKSHAK EVALUATION ==========")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("=============================================")

    print()
    print("Detailed Classification Report:")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["REAL", "FAKE"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
    )

    display.plot()

    plt.title("VaaniRakshak-AI Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200
    )

    print(
        f"Confusion matrix saved to: "
        f"{CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    evaluate()