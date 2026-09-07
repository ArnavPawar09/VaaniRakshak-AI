import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AASIST_ROOT = PROJECT_ROOT / "aasist"

sys.path.insert(0, str(AASIST_ROOT))

from models.AASIST import Model


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"AASIST device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# AASIST CONFIG
# ============================================================

MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64]
    ],
    "gat_dims": [64, 32],
    "pool_ratios": [
        0.5,
        0.7,
        0.5,
        0.5
    ],
    "temperatures": [
        2.0,
        2.0,
        100.0,
        100.0
    ],
}


TARGET_SR = 16000
TARGET_SAMPLES = 64600


# ============================================================
# LOAD AASIST
# ============================================================

_model = Model(MODEL_CONFIG)

WEIGHTS_PATH = (
    AASIST_ROOT
    / "models"
    / "weights"
    / "AASIST.pth"
)

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location="cpu"
)

_model.load_state_dict(checkpoint)

_model = _model.to(DEVICE)

_model.eval()

print("AASIST loaded successfully.")


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path):

    audio, sample_rate = sf.read(
        audio_path
    )

    # Convert stereo/multi-channel to mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    # Resample to 16 kHz
    if sample_rate != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SR
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# AUDIO PREPARATION
# ============================================================

def prepare_audio(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    ).reshape(-1)

    audio_length = len(audio)

    if audio_length == 0:

        raise ValueError(
            "Audio file contains no samples."
        )

    # --------------------------------------------------------
    # Long audio
    # --------------------------------------------------------

    if audio_length >= TARGET_SAMPLES:

        return audio[
            :TARGET_SAMPLES
        ]

    # --------------------------------------------------------
    # Short audio
    # Repeat until target length
    # --------------------------------------------------------

    repetitions = (
        TARGET_SAMPLES //
        audio_length
        + 1
    )

    padded = np.tile(
        audio,
        repetitions
    )

    return padded[
        :TARGET_SAMPLES
    ].astype(np.float32)


# ============================================================
# COMBINE AASIST FEATURES
# ============================================================

def combine_features(
    last_hidden,
    output
):

    # --------------------------------------------------------
    # last_hidden
    # Shape: [batch, 160]
    # --------------------------------------------------------

    hidden = (
        last_hidden
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # output
    # Shape: [batch, 2]
    #
    # We use the raw AASIST logits as additional learned
    # information. The VaaniRakshak classifier will learn
    # how to weight them.
    # --------------------------------------------------------

    logits = (
        output
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Combine
    #
    # [160 features] + [2 logits] = [162 features]
    # --------------------------------------------------------

    features = np.concatenate(
        [
            hidden,
            logits
        ],
        axis=1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# SINGLE FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(audio_path):

    audio = load_audio(
        audio_path
    )

    audio = prepare_audio(
        audio
    )

    # Shape:
    # [1, 64600]

    waveform = torch.from_numpy(
        audio
    ).unsqueeze(0)

    waveform = waveform.to(
        DEVICE
    )

    # AASIST
    last_hidden, output = _model(
        waveform
    )

    features = combine_features(
        last_hidden,
        output
    )

    # Return [162]
    return features[0]


# ============================================================
# BATCH FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features_batch(
    audio_paths
):

    audio_arrays = []

    for audio_path in audio_paths:

        audio = load_audio(
            audio_path
        )

        audio = prepare_audio(
            audio
        )

        audio_arrays.append(
            audio
        )

    # Shape:
    # [batch_size, 64600]

    waveform = torch.from_numpy(
        np.stack(audio_arrays)
    )

    waveform = waveform.to(
        DEVICE
    )

    # AASIST forward pass

    last_hidden, output = _model(
        waveform
    )

    features = combine_features(
        last_hidden,
        output
    )

    # Shape:
    # [batch_size, 162]

    return features


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_audio = (
        PROJECT_ROOT
        / "audio"
        / "test.wav"
    )

    features = extract_features(
        test_audio
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Feature type: {features.dtype}"
    )

    print(
        f"Expected feature count: 162"
    )