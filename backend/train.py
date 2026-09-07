import sys
import time
from pathlib import Path

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "for-norm"
    / "for-norm"
)

FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from backend.feature_extractor import extract_features_batch


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PER_CLASS = 26900
VAL_PER_CLASS = 2000
TEST_PER_CLASS = 2000

BATCH_SIZE = 16

CHECKPOINT_EVERY = 2000

RANDOM_SEED = 42

EXPECTED_FEATURES = 162

np.random.seed(RANDOM_SEED)


# ============================================================
# DATASET
# ============================================================

def get_files(split, class_name, limit):

    folder = DATASET_ROOT / split / class_name

    files = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
        ]
    )

    if len(files) < limit:

        raise ValueError(
            f"Not enough files in {folder}. "
            f"Found {len(files)}, need {limit}."
        )

    # --------------------------------------------------------
    # Deterministic random selection
    # --------------------------------------------------------

    # Do NOT use Python's hash() here because Python randomizes
    # hash values between processes.
    seed_offset = sum(
        ord(c)
        for c in f"{split}_{class_name}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED + seed_offset
    )

    selected_indices = rng.choice(
        len(files),
        size=limit,
        replace=False
    )

    return [
        files[i]
        for i in selected_indices
    ]


def build_split(split, per_class):

    real_files = get_files(
        split,
        "real",
        per_class
    )

    fake_files = get_files(
        split,
        "fake",
        per_class
    )

    paths = (
        real_files +
        fake_files
    )

    labels = (
        [0] * len(real_files) +
        [1] * len(fake_files)
    )

    # --------------------------------------------------------
    # Shuffle paths and labels together
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED + len(split)
    )

    indices = rng.permutation(
        len(paths)
    )

    paths = [
        paths[i]
        for i in indices
    ]

    labels = np.array(
        [
            labels[i]
            for i in indices
        ],
        dtype=np.int64
    )

    return paths, labels


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_split(
    split_name,
    paths,
    labels,
    checkpoint_path
):

    total = len(paths)

    X = []
    y = []

    start_index = 0

    # --------------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------------

    if checkpoint_path.exists():

        print(
            f"\nFound checkpoint: "
            f"{checkpoint_path.name}"
        )

        checkpoint = np.load(
            checkpoint_path,
            allow_pickle=False
        )

        X = list(
            checkpoint["X"]
        )

        y = list(
            checkpoint["y"]
        )

        start_index = int(
            checkpoint["next_index"]
        )

        # ----------------------------------------------------
        # Verify checkpoint feature size
        # ----------------------------------------------------

        if len(X) > 0:

            checkpoint_features = np.asarray(
                X
            )

            if (
                checkpoint_features.ndim != 2
                or
                checkpoint_features.shape[1]
                != EXPECTED_FEATURES
            ):

                raise ValueError(
                    f"Old/incompatible checkpoint detected: "
                    f"{checkpoint_features.shape}. "
                    f"Expected (*, {EXPECTED_FEATURES}). "
                    f"Delete {checkpoint_path} and restart."
                )

        print(
            f"Resuming from file "
            f"{start_index:,}/{total:,}"
        )

    else:

        print(
            f"\nStarting {split_name} extraction..."
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    start_time = time.time()

    for batch_start in range(
        start_index,
        total,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total
        )

        batch_paths = paths[
            batch_start:batch_end
        ]

        batch_labels = labels[
            batch_start:batch_end
        ]

        # ----------------------------------------------------
        # Batch extraction
        # ----------------------------------------------------

        try:

            batch_features = (
                extract_features_batch(
                    batch_paths
                )
            )

            # ------------------------------------------------
            # Verify feature dimensions
            # ------------------------------------------------

            if (
                batch_features.ndim != 2
                or
                batch_features.shape[1]
                != EXPECTED_FEATURES
            ):

                raise ValueError(
                    f"Expected batch shape "
                    f"(N, {EXPECTED_FEATURES}), "
                    f"got {batch_features.shape}"
                )

            for feature, label in zip(
                batch_features,
                batch_labels
            ):

                X.append(
                    feature
                )

                y.append(
                    label
                )

        except Exception as e:

            print(
                f"\nBatch error "
                f"{batch_start}-{batch_end}: {e}"
            )

            print(
                "Retrying this batch file-by-file..."
            )

            # ------------------------------------------------
            # Individual fallback
            # ------------------------------------------------

            for path, label in zip(
                batch_paths,
                batch_labels
            ):

                try:

                    individual_feature = (
                        extract_features_batch(
                            [path]
                        )
                    )

                    if (
                        individual_feature.ndim != 2
                        or
                        individual_feature.shape[1]
                        != EXPECTED_FEATURES
                    ):

                        raise ValueError(
                            f"Expected "
                            f"{EXPECTED_FEATURES} features, "
                            f"got "
                            f"{individual_feature.shape}"
                        )

                    X.append(
                        individual_feature[0]
                    )

                    y.append(
                        label
                    )

                except Exception as file_error:

                    print(
                        f"\nSkipped: {path.name}"
                    )

                    print(
                        f"Reason: {file_error}"
                    )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = batch_end

        elapsed = (
            time.time() -
            start_time
        )

        newly_processed = (
            processed -
            start_index
        )

        rate = (
            newly_processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total -
            processed
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else 0
        )

        eta_minutes = (
            eta_seconds / 60
        )

        print(
            f"\r{split_name}: "
            f"{processed:,}/{total:,} "
            f"({processed / total * 100:.1f}%) | "
            f"{rate:.1f} files/s | "
            f"ETA {eta_minutes:.1f} min",
            end=""
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            processed % CHECKPOINT_EVERY
            < BATCH_SIZE
            or
            processed == total
        ):

            np.savez(
                checkpoint_path,
                X=np.asarray(
                    X,
                    dtype=np.float32
                ),
                y=np.asarray(
                    y,
                    dtype=np.int64
                ),
                next_index=processed
            )

    print()

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Final shape check
    # --------------------------------------------------------

    if (
        X.ndim != 2
        or X.shape[1] != EXPECTED_FEATURES
    ):

        raise ValueError(
            f"{split_name} feature shape is "
            f"{X.shape}. "
            f"Expected "
            f"(N, {EXPECTED_FEATURES})."
        )

    print(
        f"{split_name} features: "
        f"{X.shape}"
    )

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VaaniRakshak-AI Training")
    print("162-D AASIST Representation")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nPreparing dataset..."
    )

    train_paths, train_labels = build_split(
        "training",
        TRAIN_PER_CLASS
    )

    val_paths, val_labels = build_split(
        "validation",
        VAL_PER_CLASS
    )

    test_paths, test_labels = build_split(
        "testing",
        TEST_PER_CLASS
    )

    print(
        f"Training:   "
        f"{len(train_paths):,} files"
    )

    print(
        f"Validation: "
        f"{len(val_paths):,} files"
    )

    print(
        f"Testing:    "
        f"{len(test_paths):,} files"
    )

    # --------------------------------------------------------
    # Extract training features
    # --------------------------------------------------------

    X_train, y_train = extract_split(
        "Training",
        train_paths,
        train_labels,
        FEATURES_DIR
        / "training_checkpoint.npz"
    )

    # --------------------------------------------------------
    # Extract validation features
    # --------------------------------------------------------

    X_val, y_val = extract_split(
        "Validation",
        val_paths,
        val_labels,
        FEATURES_DIR
        / "validation_checkpoint.npz"
    )

    # --------------------------------------------------------
    # Extract testing features
    # --------------------------------------------------------

    X_test, y_test = extract_split(
        "Testing",
        test_paths,
        test_labels,
        FEATURES_DIR
        / "testing_checkpoint.npz"
    )

    # --------------------------------------------------------
    # Verify all dimensions
    # --------------------------------------------------------

    print("\nFeature dimensions:")
    print(
        f"Training:   {X_train.shape}"
    )
    print(
        f"Validation: {X_val.shape}"
    )
    print(
        f"Testing:    {X_test.shape}"
    )

    if (
        X_train.shape[1] != EXPECTED_FEATURES
        or
        X_val.shape[1] != EXPECTED_FEATURES
        or
        X_test.shape[1] != EXPECTED_FEATURES
    ):

        raise ValueError(
            "Feature dimension mismatch. "
            f"Expected {EXPECTED_FEATURES}."
        )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    print(
        "\nSaving extracted features..."
    )

    np.save(
        FEATURES_DIR / "X_train.npy",
        X_train
    )

    np.save(
        FEATURES_DIR / "y_train.npy",
        y_train
    )

    np.save(
        FEATURES_DIR / "X_val.npy",
        X_val
    )

    np.save(
        FEATURES_DIR / "y_val.npy",
        y_val
    )

    np.save(
        FEATURES_DIR / "X_test.npy",
        X_test
    )

    np.save(
        FEATURES_DIR / "y_test.npy",
        y_test
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    print(
        "\nTraining classifier..."
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED
    )

    classifier.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_predictions = (
        classifier.predict(
            X_val_scaled
        )
    )

    val_probabilities = (
        classifier.predict_proba(
            X_val_scaled
        )[:, 1]
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_predictions
    )

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    print(
        "\nValidation Results"
    )

    print(
        "-" * 40
    )

    print(
        f"Accuracy:  {val_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {val_precision * 100:.2f}%"
    )

    print(
        f"Recall:    {val_recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  {val_f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   {val_auc:.4f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    test_predictions = (
        classifier.predict(
            X_test_scaled
        )
    )

    test_probabilities = (
        classifier.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print(
        "\nTest Results"
    )

    print(
        "-" * 40
    )

    print(
        f"Accuracy:  {test_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {test_precision * 100:.2f}%"
    )

    print(
        f"Recall:    {test_recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  {test_f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   {test_auc:.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    import joblib

    model_path = (
        MODELS_DIR
        / "vaani_classifier.joblib"
    )

    joblib.dump(
        {
            "scaler": scaler,

            "classifier": classifier,

            "label_map": {
                0: "REAL",
                1: "FAKE"
            },

            "feature_count": EXPECTED_FEATURES,

            "architecture": (
                "AASIST 160-D hidden features "
                "+ 2-D AASIST output logits"
            )
        },
        model_path
    )

    print(
        "\nModel saved to:"
    )

    print(
        model_path
    )

    print(
        "\nTraining complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()