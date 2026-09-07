# VaaniRakshak-AI

## AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

VaaniRakshak-AI is a deep-learning based voice authenticity detection prototype designed to distinguish between natural human speech and AI-generated or cloned speech.

The system uses **VaaniNet V3**, a custom convolutional neural network trained from scratch for synthetic speech detection.

## Features

- AI-generated and cloned voice detection
- Custom CNN trained from scratch
- Audio file upload
- Browser microphone recording
- Audio preprocessing and normalization
- 64-band log Mel spectrogram analysis
- REAL / FAKE classification
- Real and fake probability scores
- Lightweight inference model

## How It Works

```text
Audio Input
     |
     v
Audio Preprocessing
     |
     v
16 kHz Mono Audio
     |
     v
STFT + Mel Filtering
     |
     v
64-band Log Mel Spectrogram
     |
     v
VaaniNet V3 CNN
     |
     v
REAL / FAKE Classification
```

## Model

### VaaniNet V3

VaaniNet is a custom convolutional neural network developed specifically for this project.

**Architecture:**

- 4 convolutional blocks
- Batch normalization
- ReLU activation
- Max pooling
- Dropout
- Adaptive average pooling
- Fully connected classification layers
- Approximately 422K trainable parameters
- Binary classification: `REAL` / `FAKE`
- Trained from scratch

The trained V3 checkpoint is included in the repository at:

```text
models/vaaninet_v3_best.pth
```

Therefore, the original training dataset is **not required to run the application**.

## Performance

VaaniNet V3 was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 79.87% |
| F1 Score | 75.76% |
| ROC-AUC | 0.9738 |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArnavPawar09/VaaniRakshak-AI.git
cd VaaniRakshak-AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```powershell
streamlit run frontend/app.py
```

The application provides two input methods:

1. **Upload Audio** — analyze an existing audio file.
2. **Record Audio** — record a voice sample through the browser microphone and analyze it.

## Project Structure

```text
VaaniRakshak-AI/
|
├── aasist/                  # Previous/baseline anti-spoofing resources
├── audio/                   # Sample/demo audio files
├── backend/
│   ├── cnn_model.py         # VaaniNet architecture
│   ├── predict.py           # Prediction utilities
│   ├── train_cnn.py         # Earlier training version
│   ├── train_cnn_v3.py      # VaaniNet V3 training
│   ├── evaluate.py          # Model evaluation
│   └── ...
├── frontend/
│   └── app.py               # Streamlit application
├── models/
│   └── vaaninet_v3_best.pth # Final trained model
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The training dataset is not included in this repository because it is only required for model training and evaluation.

The repository contains the trained VaaniNet V3 model, allowing users to run inference without downloading or preparing the training dataset.

## Limitations

VaaniRakshak-AI is a prototype for detecting synthetic speech. A model prediction should be treated as an indication of possible voice manipulation rather than definitive proof of authenticity.

Performance can vary depending on recording conditions, audio quality, speakers, codecs, and voice-cloning systems that were not represented in the training data.

## Future Scope

- Real-time streaming voice detection
- Improved generalization to previously unseen voice-cloning systems
- Speaker verification and identity matching
- Multiple complementary anti-spoofing models
- Explainable acoustic indicators
- Integration with voice-call and communication platforms
- Continuous monitoring of incoming voice streams

## Technology Stack

- Python
- PyTorch
- Librosa
- SoundFile
- NumPy
- Streamlit
- Custom Convolutional Neural Network

## Disclaimer

This project is a research and demonstration prototype. It is intended to assist in identifying potentially synthetic or cloned speech and should not be used as the sole basis for high-stakes identity or security decisions.
