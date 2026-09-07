import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "vaaninet_best.pth"

sys.path.insert(0, str(PROJECT_ROOT))

from backend.cnn_model import VaaniNet


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = VaaniNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(audio_path):

    audio, sample_rate = sf.read(
        audio_path
    )

    # Stereo → mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    # Resample → 16 kHz
    if sample_rate != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Peak normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-6:

        audio = audio / peak

    # Fixed length
    if len(audio) >= TARGET_SAMPLES:

        audio = audio[:TARGET_SAMPLES]

    else:

        repetitions = (
            TARGET_SAMPLES // len(audio)
        ) + 1

        audio = np.tile(
            audio,
            repetitions
        )

        audio = audio[:TARGET_SAMPLES]

    return audio.astype(
        np.float32
    )


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def waveform_to_mel(waveform):

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
        spectrogram.abs() ** 2
    )

    # Mel filter bank
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

    # Log-Mel
    mel = torch.log(
        mel + 1e-6
    )

    # Standardization
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

    # Add CNN channel dimension
    mel = mel.unsqueeze(1)

    return mel


# ============================================================
# PREDICT AUDIO
# ============================================================

@torch.no_grad()
def predict_audio(audio_path):

    model = load_model()

    # --------------------------------------------
    # LOAD + PREPROCESS
    # --------------------------------------------

    audio = load_audio(
        audio_path
    )

    waveform = torch.from_numpy(
        audio
    ).unsqueeze(0)

    waveform = waveform.to(
        DEVICE
    )

    # --------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------

    mel = waveform_to_mel(
        waveform
    )

    # --------------------------------------------
    # CNN INFERENCE
    # --------------------------------------------

    logits = model(
        mel
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    # Class mapping:
    # 0 = REAL
    # 1 = FAKE

    real_probability = (
        probabilities[0].item() * 100
    )

    fake_probability = (
        probabilities[1].item() * 100
    )

    # --------------------------------------------
    # FINAL CLASSIFICATION
    # --------------------------------------------

    if fake_probability >= real_probability:

        prediction = "FAKE"
        confidence = fake_probability

    else:

        prediction = "REAL"
        confidence = real_probability

    return (
        prediction,
        confidence,
        real_probability,
        fake_probability
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python backend/predict.py <audio_file>\n"
        )

        print(
            "Example:"
        )

        print(
            "python backend/predict.py audio/test_fake.wav\n"
        )

        sys.exit(1)


    audio_path = Path(
        sys.argv[1]
    )


    if not audio_path.exists():

        print(
            f"\nError: Audio file not found:\n{audio_path}"
        )

        sys.exit(1)


    print(
        "\n========================================"
    )

    print(
        "        VaaniRakshak-AI"
    )

    print(
        "        VaaniNet Detector"
    )

    print(
        "========================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Audio:  {audio_path}"
    )

    print(
        "\nAnalyzing..."
    )


    try:

        (
            prediction,
            confidence,
            real_probability,
            fake_probability
        ) = predict_audio(
            audio_path
        )


        print(
            "\n----------------------------------------"
        )

        print(
            f"Prediction : {prediction}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )

        print(
            f"Real       : {real_probability:.2f}%"
        )

        print(
            f"Fake       : {fake_probability:.2f}%"
        )

        print(
            "----------------------------------------"
        )


        if prediction == "REAL":

            print(
                "\n✓ The audio appears to be genuine human speech."
            )

        else:

            print(
                "\n⚠ The audio appears to contain synthetic or cloned speech."
            )


        print(
            "\n========================================\n"
        )


    except Exception as e:

        print(
            f"\nPrediction failed: {e}\n"
        )

        sys.exit(1)