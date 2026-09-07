import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vaaninet_v2_best.pth"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from backend.cnn_model import VaaniNet

from backend.train_cnn import (
    build_split,
    VoiceDataset,
    waveform_to_mel,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("VaaniNet V2 - Threshold Analysis")
print("=" * 65)

print()

print(
    f"Device: {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()

model = VaaniNet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Model: {MODEL_PATH.name}"
)


# ============================================================
# GET DATA
# ============================================================

print(
    "\nPreparing validation set..."
)

val_paths, val_labels = build_split(
    "validation",
    None
)

print(
    f"Validation files: {len(val_paths):,}"
)

val_dataset = VoiceDataset(
    val_paths,
    val_labels,
    training=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# COLLECT VALIDATION PROBABILITIES
# ============================================================

@torch.no_grad()
def collect_predictions(loader):

    labels = []

    probabilities = []

    for batch_index, (
        waveforms,
        batch_labels
    ) in enumerate(
        loader,
        start=1
    ):

        waveforms = waveforms.to(
            DEVICE,
            non_blocking=True
        )

        mel = waveform_to_mel(
            waveforms
        )

        logits = model(
            mel
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        fake_probs = probs[
            :,
            1
        ]

        labels.extend(
            batch_labels.numpy()
        )

        probabilities.extend(
            fake_probs.cpu().numpy()
        )

        print(
            f"\rValidation batch "
            f"{batch_index:,}/"
            f"{len(loader):,}",
            end=""
        )

    print()

    return (
        np.array(labels),
        np.array(probabilities)
    )


print(
    "\nCollecting validation predictions..."
)

val_y, val_probs = collect_predictions(
    val_loader
)


# ============================================================
# ROC-AUC
# ============================================================

val_auc = roc_auc_score(
    val_y,
    val_probs
)

print(
    f"\nValidation ROC-AUC: "
    f"{val_auc:.4f}"
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print(
    "\nSearching for optimal threshold..."
)

best_threshold = 0.50

best_f1 = -1.0

best_accuracy = 0.0

best_precision = 0.0

best_recall = 0.0

results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

for threshold in thresholds:

    predictions = (
        val_probs >= threshold
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        val_y,
        predictions
    )

    precision = precision_score(
        val_y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_y,
        predictions,
        zero_division=0
    )

    results.append(
        (
            threshold,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_accuracy = accuracy

        best_precision = precision

        best_recall = recall


# ============================================================
# SHOW TOP THRESHOLDS
# ============================================================

results.sort(
    key=lambda x: x[4],
    reverse=True
)

print(
    "\nTop 10 validation thresholds"
)

print(
    "-" * 75
)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>12} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'F1':>12}"
)

print(
    "-" * 75
)

for row in results[:10]:

    threshold, accuracy, precision, recall, f1 = row

    print(
        f"{threshold:10.2f} "
        f"{accuracy * 100:11.2f}% "
        f"{precision * 100:11.2f}% "
        f"{recall * 100:11.2f}% "
        f"{f1 * 100:11.2f}%"
    )


# ============================================================
# BEST VALIDATION THRESHOLD
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "BEST VALIDATION THRESHOLD"
)

print(
    "=" * 65
)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_accuracy * 100:.2f}%"
)

print(
    f"Precision : {best_precision * 100:.2f}%"
)

print(
    f"Recall    : {best_recall * 100:.2f}%"
)

print(
    f"F1        : {best_f1 * 100:.2f}%"
)


# ============================================================
# TEST SET
# ============================================================

print(
    "\nPreparing untouched test set..."
)

test_paths, test_labels = build_split(
    "testing",
    None
)

print(
    f"Test files: {len(test_paths):,}"
)

test_dataset = VoiceDataset(
    test_paths,
    test_labels,
    training=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)

print(
    "\nCollecting test predictions..."
)

test_y, test_probs = collect_predictions(
    test_loader
)


# ============================================================
# TEST AT DEFAULT 0.50
# ============================================================

default_predictions = (
    test_probs >= 0.50
).astype(
    np.int64
)

default_accuracy = accuracy_score(
    test_y,
    default_predictions
)

default_precision = precision_score(
    test_y,
    default_predictions,
    zero_division=0
)

default_recall = recall_score(
    test_y,
    default_predictions,
    zero_division=0
)

default_f1 = f1_score(
    test_y,
    default_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_y,
    test_probs
)


# ============================================================
# TEST AT OPTIMAL VALIDATION THRESHOLD
# ============================================================

optimized_predictions = (
    test_probs >= best_threshold
).astype(
    np.int64
)

optimized_accuracy = accuracy_score(
    test_y,
    optimized_predictions
)

optimized_precision = precision_score(
    test_y,
    optimized_predictions,
    zero_division=0
)

optimized_recall = recall_score(
    test_y,
    optimized_predictions,
    zero_division=0
)

optimized_f1 = f1_score(
    test_y,
    optimized_predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_y,
    optimized_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 65
)

print(
    f"ROC-AUC: {test_auc:.4f}"
)

print()

print(
    "Default threshold = 0.50"
)

print(
    f"Accuracy : {default_accuracy * 100:.2f}%"
)

print(
    f"Precision: {default_precision * 100:.2f}%"
)

print(
    f"Recall   : {default_recall * 100:.2f}%"
)

print(
    f"F1       : {default_f1 * 100:.2f}%"
)

print()

print(
    f"Optimized threshold = {best_threshold:.2f}"
)

print(
    f"Accuracy : {optimized_accuracy * 100:.2f}%"
)

print(
    f"Precision: {optimized_precision * 100:.2f}%"
)

print(
    f"Recall   : {optimized_recall * 100:.2f}%"
)

print(
    f"F1       : {optimized_f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix"
)

print(
    "(rows = actual, columns = predicted)"
)

print()

print(
    "              REAL    FAKE"
)

print(
    f"REAL       {cm[0,0]:6d}  {cm[0,1]:6d}"
)

print(
    f"FAKE       {cm[1,0]:6d}  {cm[1,1]:6d}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "THRESHOLD IMPROVEMENT"
)

print(
    "=" * 65
)

print(
    f"Accuracy change: "
    f"{(optimized_accuracy - default_accuracy) * 100:+.2f} pp"
)

print(
    f"F1 change: "
    f"{(optimized_f1 - default_f1) * 100:+.2f} pp"
)

print(
    f"Recall change: "
    f"{(optimized_recall - default_recall) * 100:+.2f} pp"
)

print(
    f"Precision change: "
    f"{(optimized_precision - default_precision) * 100:+.2f} pp"
)

print(
    "\nThreshold analysis complete."
)