import sys
from pathlib import Path

import torch
import soundfile as sf

# Add the AASIST repository to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AASIST_DIR = PROJECT_ROOT / "aasist"

sys.path.insert(0, str(AASIST_DIR))

from models.AASIST import Model


# AASIST configuration
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

MODEL_PATH = AASIST_DIR / "models" / "weights" / "AASIST.pth"


def load_model():
    print("Loading AASIST model...")

    model = Model(MODEL_CONFIG)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("Model loaded successfully.")

    return model


def load_audio(audio_path):
    audio, sample_rate = sf.read(audio_path)

    print(f"Sample rate: {sample_rate}")
    print(f"Samples: {len(audio)}")

    # Convert stereo → mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    return audio, sample_rate


def prepare_audio(audio, sample_rate):
    TARGET_SAMPLE_RATE = 16000
    TARGET_SAMPLES = 64600

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected 16 kHz audio, but received {sample_rate} Hz."
        )

    # Convert to torch tensor
    audio = torch.tensor(audio, dtype=torch.float32)

    # Make audio exactly 64,600 samples
    if len(audio) < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - len(audio)
        audio = torch.nn.functional.pad(audio, (0, padding))

    else:
        audio = audio[:TARGET_SAMPLES]

    # Add batch dimension
    audio = audio.unsqueeze(0)

    return audio


def predict(model, audio):
    with torch.no_grad():
        _, output = model(audio)

        probabilities = torch.softmax(output, dim=1)

    fake_probability = probabilities[0][1].item()
    real_probability = probabilities[0][0].item()

    return real_probability, fake_probability


def main():
    model = load_model()

    # We'll replace this with an actual audio file shortly
    audio_path = PROJECT_ROOT / "audio" / "assistant.wav"

    audio, sample_rate = load_audio(audio_path)

    audio = prepare_audio(audio, sample_rate)

    real_probability, fake_probability = predict(
        model,
        audio
    )

    print()
    print("========== VaaniRakshak ==========")
    print(f"Real probability: {real_probability * 100:.2f}%")
    print(f"Fake probability: {fake_probability * 100:.2f}%")

    if fake_probability >= 0.5:
        print("Prediction: FAKE / SPOOF")
    else:
        print("Prediction: REAL / BONAFIDE")

    print("==================================")


if __name__ == "__main__":
    main()