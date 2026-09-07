import sys
import time
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(
    exist_ok=True
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from backend.cnn_model import VaaniNet


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SAMPLE_RATE = 16000

# 4 seconds
TARGET_SAMPLES = 64600

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.0004

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# Use all available files
TRAIN_PER_CLASS = None
VAL_PER_CLASS = None
TEST_PER_CLASS = None

# Early stopping
EARLY_STOPPING_PATIENCE = 4

# ============================================================
# V3 CLASS WEIGHTS
#
# Class 0 = REAL
# Class 1 = FAKE
#
# Fake errors receive a higher penalty.
# ============================================================

REAL_WEIGHT = 1.0
FAKE_WEIGHT = 1.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("VaaniRakshak-AI - Custom VaaniNet V3")
print("=" * 65)

print()

print(
    f"Device: {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# AUDIO VALIDATION
# ============================================================

def is_valid_audio(path):

    try:

        info = sf.info(
            str(path)
        )

        if info.frames <= 0:

            return False

        if info.samplerate <= 0:

            return False

        return True

    except Exception:

        return False


# ============================================================
# FILE SELECTION
# ============================================================

def get_files(
    split,
    class_name,
    limit=None
):

    folder = (
        DATASET_ROOT
        / split
        / class_name
    )

    if not folder.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    all_files = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
        ]
    )

    valid_files = []

    invalid_count = 0

    for path in all_files:

        if is_valid_audio(path):

            valid_files.append(
                path
            )

        else:

            invalid_count += 1

    print(
        f"{split:10s} | "
        f"{class_name:5s} | "
        f"valid={len(valid_files):5d} | "
        f"invalid={invalid_count:3d}"
    )

    if len(valid_files) == 0:

        raise ValueError(
            f"No valid audio files found in {folder}"
        )

    if limit is None:

        return valid_files

    if len(valid_files) < limit:

        raise ValueError(
            f"Not enough valid files in {folder}. "
            f"Found {len(valid_files)}, "
            f"need {limit}."
        )

    rng = np.random.default_rng(
        SEED
        + sum(
            ord(c)
            for c in split + class_name
        )
    )

    indices = rng.choice(
        len(valid_files),
        size=limit,
        replace=False
    )

    return [
        valid_files[i]
        for i in indices
    ]


def build_split(
    split,
    per_class=None
):

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
        real_files
        + fake_files
    )

    labels = (
        [0] * len(real_files)
        +
        [1] * len(fake_files)
    )

    rng = np.random.default_rng(
        SEED
        + len(split)
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

    return (
        paths,
        labels
    )


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(
    audio_path,
    training=False
):

    audio, sample_rate = sf.read(
        audio_path
    )

    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    audio = (
        audio
        - np.mean(audio)
    )

    # --------------------------------------------------------
    # Peak normalization
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-6:

        audio = (
            audio / peak
        )

    # --------------------------------------------------------
    # Fixed-length audio
    # --------------------------------------------------------

    if len(audio) >= TARGET_SAMPLES:

        if training:

            max_start = (
                len(audio)
                - TARGET_SAMPLES
            )

            if max_start > 0:

                start = random.randint(
                    0,
                    max_start
                )

                audio = audio[
                    start:
                    start + TARGET_SAMPLES
                ]

            else:

                audio = audio[
                    :TARGET_SAMPLES
                ]

        else:

            audio = audio[
                :TARGET_SAMPLES
            ]

    else:

        # ----------------------------------------------------
        # Reflection padding for short audio.
        # Avoids periodic tiling of the recording.
        # ----------------------------------------------------

        missing = (
            TARGET_SAMPLES
            - len(audio)
        )

        if len(audio) > 1:

            if training:

                left = random.randint(
                    0,
                    missing
                )

                right = (
                    missing
                    - left
                )

                audio = np.pad(
                    audio,
                    (
                        left,
                        right
                    ),
                    mode="reflect"
                )

            else:

                left = missing // 2

                right = (
                    missing
                    - left
                )

                audio = np.pad(
                    audio,
                    (
                        left,
                        right
                    ),
                    mode="reflect"
                )

        else:

            audio = np.pad(
                audio,
                (
                    0,
                    missing
                ),
                mode="constant"
            )

    return audio.astype(
        np.float32
    )


# ============================================================
# WAVEFORM AUGMENTATION
# ============================================================

def augment_waveform(
    audio
):

    # --------------------------------------------------------
    # Random gain
    # --------------------------------------------------------

    if random.random() < 0.6:

        gain = random.uniform(
            0.70,
            1.20
        )

        audio = (
            audio
            * gain
        )

    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------

    if random.random() < 0.35:

        noise_level = random.uniform(
            0.001,
            0.006
        )

        noise = np.random.normal(
            0,
            noise_level,
            size=audio.shape
        ).astype(
            np.float32
        )

        audio = (
            audio
            + noise
        )

    # --------------------------------------------------------
    # Small time shift
    # --------------------------------------------------------

    if random.random() < 0.35:

        shift = random.randint(
            -1200,
            1200
        )

        audio = np.roll(
            audio,
            shift
        )

    # --------------------------------------------------------
    # Random polarity inversion
    # --------------------------------------------------------

    if random.random() < 0.10:

        audio = -audio

    # --------------------------------------------------------
    # Keep waveform stable
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1.0:

        audio = (
            audio / peak
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class VoiceDataset(
    Dataset
):

    def __init__(
        self,
        paths,
        labels,
        training=False
    ):

        self.paths = paths

        self.labels = labels

        self.training = training

    def __len__(
        self
    ):

        return len(
            self.paths
        )

    def __getitem__(
        self,
        index
    ):

        path = self.paths[index]

        label = int(
            self.labels[index]
        )

        audio = load_audio(
            path,
            training=self.training
        )

        if self.training:

            audio = augment_waveform(
                audio
            )

        return (
            torch.from_numpy(
                audio
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def waveform_to_mel(
    waveform
):

    """
    Input:
        [batch, samples]

    Output:
        [batch, 1, 64, time]
    """

    window = torch.hann_window(
        N_FFT,
        device=waveform.device
    )

    spectrogram = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True
    )

    # Power spectrogram
    magnitude = (
        spectrogram.abs()
        ** 2
    )

    # --------------------------------------------------------
    # Mel filter bank
    # --------------------------------------------------------

    mel_filter = torch.tensor(
        librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=20,
            fmax=8000
        ),
        dtype=torch.float32,
        device=waveform.device
    )

    mel = torch.matmul(
        mel_filter,
        magnitude
    )

    # --------------------------------------------------------
    # Log-Mel
    # --------------------------------------------------------

    mel = torch.log(
        mel + 1e-6
    )

    # --------------------------------------------------------
    # Per-spectrogram normalization
    # --------------------------------------------------------

    mean = mel.mean(
        dim=(1, 2),
        keepdim=True
    )

    std = mel.std(
        dim=(1, 2),
        keepdim=True
    )

    mel = (
        mel - mean
    ) / (
        std + 1e-6
    )

    # --------------------------------------------------------
    # CNN channel dimension
    # --------------------------------------------------------

    mel = mel.unsqueeze(1)

    return mel


# ============================================================
# SPECAUGMENT
# ============================================================

def spec_augment(
    mel
):

    """
    Input:
        [B, 1, M, T]

    Randomly masks portions of the frequency and
    time dimensions during training.
    """

    augmented = mel.clone()

    batch_size = augmented.size(0)

    mel_bins = augmented.size(2)

    time_steps = augmented.size(3)

    for i in range(
        batch_size
    ):

        # ----------------------------------------------------
        # Frequency masking
        # ----------------------------------------------------

        if random.random() < 0.7:

            max_width = max(
                1,
                int(
                    mel_bins * 0.15
                )
            )

            width = random.randint(
                1,
                max_width
            )

            if width < mel_bins:

                start = random.randint(
                    0,
                    mel_bins - width
                )

                augmented[
                    i,
                    :,
                    start:
                    start + width,
                    :
                ] = 0

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        if random.random() < 0.7:

            max_width = max(
                1,
                int(
                    time_steps * 0.10
                )
            )

            width = random.randint(
                1,
                max_width
            )

            if width < time_steps:

                start = random.randint(
                    0,
                    time_steps - width
                )

                augmented[
                    i,
                    :,
                    :,
                    start:
                    start + width
                ] = 0

    return augmented


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch
):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    start_time = time.time()

    total_batches = len(
        loader
    )

    for batch_index, (
        waveforms,
        labels
    ) in enumerate(
        loader,
        start=1
    ):

        waveforms = waveforms.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Waveform -> Mel
        # ----------------------------------------------------

        mel = waveform_to_mel(
            waveforms
        )

        # ----------------------------------------------------
        # SpecAugment
        # ----------------------------------------------------

        mel = spec_augment(
            mel
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            mel
        )

        loss = criterion(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            batch_index
            / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total_batches
            - batch_index
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"\rEpoch {epoch:02d} | "
            f"Batch "
            f"{batch_index:,}/"
            f"{total_batches:,} | "
            f"Loss "
            f"{loss.item():.4f} | "
            f"ETA "
            f"{eta / 60:.1f} min",
            end=""
        )

    print()

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    average_loss = (
        running_loss
        / total_batches
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    all_labels = []

    all_predictions = []

    all_probabilities = []

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    for waveforms, labels in loader:

        waveforms = waveforms.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        mel = waveform_to_mel(
            waveforms
        )

        logits = model(
            mel
        )

        loss = criterion(
            logits,
            labels
        )

        total_loss += (
            loss.item()
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[
                :,
                1
            ]
            .cpu()
            .numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    average_loss = (
        total_loss
        / len(loader)
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\nPreparing dataset..."
    )

    print(
        "\nValidating audio files:"
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

    print()

    print(
        f"Training files:   {len(train_paths):,}"
    )

    print(
        f"Validation files: {len(val_paths):,}"
    )

    print(
        f"Testing files:    {len(test_paths):,}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = VoiceDataset(
        train_paths,
        train_labels,
        training=True
    )

    val_dataset = VoiceDataset(
        val_paths,
        val_labels,
        training=False
    )

    test_dataset = VoiceDataset(
        test_paths,
        test_labels,
        training=False
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nCreating VaaniNet..."
    )

    model = VaaniNet().to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_parameters:,}"
    )

    # ========================================================
    # V3 CLASS-WEIGHTED LOSS
    # ========================================================

    class_weights = torch.tensor(
        [
            REAL_WEIGHT,
            FAKE_WEIGHT
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    print(
        "\nClass weights:"
    )

    print(
        f"REAL: {REAL_WEIGHT}"
    )

    print(
        f"FAKE: {FAKE_WEIGHT}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_f1 = -1.0

    best_auc = -1.0

    epochs_without_improvement = 0

    best_model_path = (
        MODELS_DIR
        / "vaaninet_v3_best.pth"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\nStarting VaaniNet V3 training..."
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Fake class weight: {FAKE_WEIGHT}"
    )

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "=" * 65
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                epoch
            )
        )

        validation = evaluate(
            model,
            val_loader
        )

        scheduler.step(
            validation["f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()

        print(
            f"Epoch {epoch}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation['loss']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation['accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Precision: "
            f"{validation['precision'] * 100:.2f}%"
        )

        print(
            f"Validation Recall: "
            f"{validation['recall'] * 100:.2f}%"
        )

        print(
            f"Validation F1: "
            f"{validation['f1'] * 100:.2f}%"
        )

        print(
            f"Validation ROC-AUC: "
            f"{validation['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        improved = False

        if validation["f1"] > best_f1:

            improved = True

        elif (
            validation["f1"] == best_f1
            and validation["roc_auc"] > best_auc
        ):

            improved = True

        if improved:

            best_f1 = validation[
                "f1"
            ]

            best_auc = validation[
                "roc_auc"
            ]

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "sample_rate":
                        SAMPLE_RATE,

                    "target_samples":
                        TARGET_SAMPLES,

                    "n_mels":
                        N_MELS,

                    "n_fft":
                        N_FFT,

                    "hop_length":
                        HOP_LENGTH,

                    "label_map": {
                        0: "REAL",
                        1: "FAKE"
                    },

                    "version":
                        "VaaniNet V3",

                    "class_weights": {
                        "REAL":
                            REAL_WEIGHT,

                        "FAKE":
                            FAKE_WEIGHT
                    }
                },
                best_model_path
            )

            print(
                "\n✓ New best V3 model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "Loading best VaaniNet V3 model..."
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\nFinal Test Results"
    )

    print(
        "-" * 45
    )

    test_results = evaluate(
        model,
        test_loader
    )

    print(
        f"Accuracy:  "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{test_results['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{test_results['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:  "
        f"{test_results['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   "
        f"{test_results['roc_auc']:.4f}"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "VaaniNet Model Comparison"
    )

    print(
        "=" * 65
    )

    print()

    print(
        "V1"
    )

    print(
        "  Accuracy : 72.88%"
    )

    print(
        "  F1       : 62.91%"
    )

    print(
        "  ROC-AUC  : 0.9715"
    )

    print()

    print(
        "V2"
    )

    print(
        "  Accuracy : 76.44%"
    )

    print(
        "  F1       : 70.13%"
    )

    print(
        "  ROC-AUC  : 0.9834"
    )

    print()

    print(
        "V3"
    )

    print(
        f"  Accuracy : "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"  F1       : "
        f"{test_results['f1'] * 100:.2f}%"
    )

    print(
        f"  ROC-AUC  : "
        f"{test_results['roc_auc']:.4f}"
    )

    print()

    # ========================================================
    # CHANGES FROM V2
    # ========================================================

    accuracy_change = (
        test_results["accuracy"]
        - 0.7644
    ) * 100

    f1_change = (
        test_results["f1"]
        - 0.7013
    ) * 100

    auc_change = (
        test_results["roc_auc"]
        - 0.9834
    )

    print(
        "V3 vs V2"
    )

    print(
        f"  Accuracy change: "
        f"{accuracy_change:+.2f} pp"
    )

    print(
        f"  F1 change: "
        f"{f1_change:+.2f} pp"
    )

    print(
        f"  ROC-AUC change: "
        f"{auc_change:+.4f}"
    )

    # ========================================================
    # MODEL PATH
    # ========================================================

    print(
        "\nBest model:"
    )

    print(
        best_model_path
    )

    print(
        "\nVaaniNet V3 training complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()