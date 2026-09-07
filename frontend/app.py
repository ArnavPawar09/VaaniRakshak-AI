import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.cnn_model import VaaniNet


MODEL_PATH = PROJECT_ROOT / "models" / "vaaninet_v3_best.pth"
AUDIO_DIR = PROJECT_ROOT / "audio"

AUDIO_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VaaniRakshak-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.hero {
    padding: 10px 0 28px 0;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(49, 51, 63, 0.08);
    border: 1px solid rgba(49, 51, 63, 0.14);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-subtitle {
    font-size: 16px;
    line-height: 1.6;
    opacity: 0.65;
    max-width: 760px;
    margin-top: 12px;
}


/* ============================================================
   INFO CARDS
   ============================================================ */

.info-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
    background: rgba(128,128,128,0.025);
}

.info-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 7px;
}

.info-text {
    font-size: 13px;
    line-height: 1.55;
    opacity: 0.65;
}


/* ============================================================
   INPUT AREA
   ============================================================ */

.section-title {
    font-size: 23px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-description {
    font-size: 14px;
    opacity: 0.6;
    margin-bottom: 18px;
}

.input-card {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 16px;
    padding: 22px;
    min-height: 170px;
    margin-bottom: 14px;
}

.input-card-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
}

.input-card-description {
    font-size: 13px;
    opacity: 0.6;
    margin-bottom: 18px;
}


/* ============================================================
   RESULT
   ============================================================ */

.result-wrapper {
    margin-top: 32px;
}

.result-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}

.result-icon {
    font-size: 38px;
    margin-bottom: 8px;
}

.result-label {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.7px;
}

.result-confidence {
    font-size: 17px;
    margin-top: 8px;
    opacity: 0.7;
}

.result-description {
    max-width: 650px;
    margin: 18px auto 0 auto;
    font-size: 14px;
    line-height: 1.6;
    opacity: 0.65;
}


/* ============================================================
   PROBABILITY / METRIC CARDS
   ============================================================ */

.prob-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.prob-label {
    font-size: 12px;
    font-weight: 600;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.prob-value {
    font-size: 25px;
    font-weight: 750;
    margin-top: 4px;
}


/* ============================================================
   DIAGNOSTICS
   ============================================================ */

.diagnostic-box {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 14px;
    font-size: 12px;
    line-height: 1.7;
    opacity: 0.7;
}

.diagnostic-title {
    font-weight: 700;
    opacity: 1;
    margin-bottom: 3px;
}


/* ============================================================
   HOW IT WORKS
   ============================================================ */

.workflow-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
    background: rgba(128,128,128,0.025);
}

.workflow-number {
    font-size: 12px;
    font-weight: 700;
    opacity: 0.5;
    margin-bottom: 6px;
}

.workflow-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}

.workflow-text {
    font-size: 13px;
    line-height: 1.55;
    opacity: 0.65;
}


/* ============================================================
   MODEL DETAILS
   ============================================================ */

.model-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 16px 18px;
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.6;
    opacity: 0.75;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    font-size: 12px;
    opacity: 0.45;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid rgba(128,128,128,0.15);
}

</style>
""")


# ============================================================
# MODEL SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TARGET_SR = 16000
TARGET_SAMPLES = 64600

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

FMIN = 20
FMAX = 8000


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = VaaniNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load VaaniNet: {e}"
    )

    st.stop()


# ============================================================
# AUDIO INFO
# ============================================================

def get_audio_info(path):

    try:

        info = sf.info(path)

        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# AUDIO CANONICALIZATION
# ============================================================

def canonicalize_audio(
    input_path,
    output_path
):

    audio, sr = sf.read(
        input_path,
        always_2d=False
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if sr != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    if len(audio) > 0:

        audio = (
            audio
            - np.mean(audio)
        )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    sf.write(
        output_path,
        audio,
        TARGET_SR,
        subtype="PCM_16"
    )

    return output_path


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):

    audio, sr = sf.read(
        audio_path,
        always_2d=False
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    if sr != TARGET_SR:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    audio = (
        audio
        - np.mean(audio)
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    if len(audio) >= TARGET_SAMPLES:

        audio = audio[
            :TARGET_SAMPLES
        ]

    else:

        pad_amount = (
            TARGET_SAMPLES
            - len(audio)
        )

        if len(audio) > 1:

            audio = np.pad(
                audio,
                (0, pad_amount),
                mode="reflect"
            )

        else:

            audio = np.pad(
                audio,
                (0, pad_amount),
                mode="constant"
            )

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    )

    window = torch.hann_window(
        N_FFT
    )

    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True
    )

    power = stft.abs().pow(2)

    mel_filter = librosa.filters.mel(
        sr=TARGET_SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    mel_filter = torch.tensor(
        mel_filter,
        dtype=torch.float32
    )

    mel = torch.matmul(
        mel_filter,
        power
    )

    mel = torch.log(
        mel + 1e-6
    )

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-6
    )

    mel = mel.unsqueeze(0)
    mel = mel.unsqueeze(0)

    return mel


# ============================================================
# PREDICTION
# ============================================================

def predict_audio(audio_path):

    x = preprocess_audio(
        audio_path
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    real_probability = (
        probabilities[0].item()
    )

    fake_probability = (
        probabilities[1].item()
    )

    if fake_probability >= real_probability:

        label = "FAKE"
        confidence = fake_probability

    else:

        label = "REAL"
        confidence = real_probability

    return (
        label,
        confidence,
        real_probability,
        fake_probability
    )


# ============================================================
# SAVE + CANONICALIZE INPUT
# ============================================================

def save_uploaded_audio(
    audio_file,
    prefix
):

    original_name = getattr(
        audio_file,
        "name",
        "audio.wav"
    )

    original_type = getattr(
        audio_file,
        "type",
        "unknown"
    )

    suffix = Path(
        original_name
    ).suffix.lower()

    if not suffix:

        suffix = ".wav"

    original_path = (
        AUDIO_DIR
        / f"{prefix}_original{suffix}"
    )

    audio_bytes = (
        audio_file.getvalue()
    )

    with open(
        original_path,
        "wb"
    ) as f:

        f.write(audio_bytes)

    original_info = get_audio_info(
        original_path
    )

    if "error" in original_info:

        st.error(
            "Unable to decode the audio: "
            + original_info["error"]
        )

        return None

    st.html(f"""
    <div class="diagnostic-box">

        <div class="diagnostic-title">
            Audio Information
        </div>

        File: {original_name}<br>
        MIME: {original_type}<br>
        Size: {len(audio_bytes):,} bytes<br>
        Sample rate: {original_info['samplerate']} Hz<br>
        Channels: {original_info['channels']}<br>
        Duration: {original_info['duration']:.2f} sec<br>
        Format: {original_info['format']} /
        {original_info['subtype']}

    </div>
    """)

    canonical_path = (
        AUDIO_DIR
        / f"{prefix}_canonical.wav"
    )

    try:

        canonicalize_audio(
            original_path,
            canonical_path
        )

    except Exception as e:

        st.error(
            f"Audio conversion failed: {e}"
        )

        return None

    return canonical_path


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    label,
    confidence,
    real_probability,
    fake_probability
):

    if label == "FAKE":

        st.html(f"""
        <div class="result-card">

            <div class="result-icon">
                ⚠️
            </div>

            <div class="result-label">
                AI-GENERATED VOICE DETECTED
            </div>

            <div class="result-confidence">
                Detection confidence:
                <b>{confidence * 100:.2f}%</b>
            </div>

            <div class="result-description">
                The acoustic characteristics of this sample
                are more consistent with synthetic or
                cloned speech than natural human speech.
            </div>

        </div>
        """)

    else:

        st.html(f"""
        <div class="result-card">

            <div class="result-icon">
                ✓
            </div>

            <div class="result-label">
                HUMAN VOICE DETECTED
            </div>

            <div class="result-confidence">
                Detection confidence:
                <b>{confidence * 100:.2f}%</b>
            </div>

            <div class="result-description">
                The acoustic characteristics of this sample
                are more consistent with natural human speech.
            </div>

        </div>
        """)

    st.write("")

    prob1, prob2 = st.columns(2)

    with prob1:

        st.html(f"""
        <div class="prob-card">

            <div class="prob-label">
                Real Probability
            </div>

            <div class="prob-value">
                {real_probability * 100:.2f}%
            </div>

        </div>
        """)

    with prob2:

        st.html(f"""
        <div class="prob-card">

            <div class="prob-label">
                Fake Probability
            </div>

            <div class="prob-value">
                {fake_probability * 100:.2f}%
            </div>

        </div>
        """)


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        AI VOICE AUTHENTICITY DETECTION
    </div>

    <h1 class="hero-title">
        VaaniRakshak-AI
    </h1>

    <div class="hero-subtitle">
        Detect AI-generated and cloned voices using
        a custom deep-learning model trained specifically
        for synthetic speech detection.
    </div>

</div>
""")


# ============================================================
# INFO CARDS
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            Custom AI Model
        </div>

        <div class="info-text">
            VaaniNet is a custom convolutional neural
            network trained from scratch for voice
            authenticity classification.
        </div>

    </div>
    """)

with info2:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            Acoustic Analysis
        </div>

        <div class="info-text">
            Audio is converted into a normalized
            Mel spectrogram before being analyzed
            by the neural network.
        </div>

    </div>
    """)

with info3:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            Two Input Methods
        </div>

        <div class="info-text">
            Analyze an existing audio file or record
            a voice sample directly through the
            browser microphone.
        </div>

    </div>
    """)


# ============================================================
# INPUT SECTION
# ============================================================

st.html("""
<div class="section-title">
    Analyze Voice
</div>

<div class="section-description">
    Choose an input method below and analyze the voice sample.
</div>
""")


tab_upload, tab_record = st.tabs(
    [
        "📁  Upload Audio",
        "🎙️  Record Audio"
    ]
)


# ============================================================
# UPLOAD TAB
# ============================================================

with tab_upload:

    st.html("""
    <div class="input-card">

        <div class="input-card-title">
            Upload an audio file
        </div>

        <div class="input-card-description">
            Supported formats: WAV, MP3, FLAC, OGG and M4A.
        </div>

    </div>
    """)

    uploaded_file = st.file_uploader(
        "Select audio",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        st.audio(
            uploaded_file
        )

        if st.button(
            "Analyze Uploaded Audio",
            type="primary",
            use_container_width=True,
            key="analyze_upload"
        ):

            with st.spinner(
                "Analyzing voice..."
            ):

                audio_path = (
                    save_uploaded_audio(
                        uploaded_file,
                        "_frontend_upload"
                    )
                )

                if audio_path is not None:

                    try:

                        result = predict_audio(
                            audio_path
                        )

                        st.session_state[
                            "prediction"
                        ] = result

                    except Exception as e:

                        st.error(
                            f"Prediction failed: {e}"
                        )


# ============================================================
# RECORD TAB
# ============================================================

with tab_record:

    st.html("""
    <div class="input-card">

        <div class="input-card-title">
            Record a voice sample
        </div>

        <div class="input-card-description">
            Record through your microphone, stop the recording,
            then analyze the captured audio.
        </div>

    </div>
    """)

    recorded_audio = st.audio_input(
        "Record voice",
        label_visibility="collapsed"
    )

    if recorded_audio is not None:

        st.audio(
            recorded_audio
        )

        if st.button(
            "Analyze Recording",
            type="primary",
            use_container_width=True,
            key="analyze_recording"
        ):

            with st.spinner(
                "Analyzing recorded voice..."
            ):

                audio_path = (
                    save_uploaded_audio(
                        recorded_audio,
                        "_frontend_recording"
                    )
                )

                if audio_path is not None:

                    try:

                        result = predict_audio(
                            audio_path
                        )

                        st.session_state[
                            "prediction"
                        ] = result

                    except Exception as e:

                        st.error(
                            f"Prediction failed: {e}"
                        )


# ============================================================
# DETECTION RESULT
# ============================================================

if "prediction" in st.session_state:

    (
        label,
        confidence,
        real_probability,
        fake_probability
    ) = st.session_state["prediction"]

    st.divider()

    display_result(
        label,
        confidence,
        real_probability,
        fake_probability
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.html("""
<div class="section-title">
    How It Works
</div>

<div class="section-description">
    VaaniRakshak-AI transforms speech into an acoustic
    representation and uses VaaniNet to determine whether
    the voice is human or synthetic.
</div>
""")


step1, step2, step3, step4 = st.columns(4)

with step1:

    st.html("""
    <div class="workflow-card">

        <div class="workflow-number">
            01 · INPUT
        </div>

        <div class="workflow-title">
            Audio Input
        </div>

        <div class="workflow-text">
            Upload an audio file or record a voice sample
            using the microphone.
        </div>

    </div>
    """)


with step2:

    st.html("""
    <div class="workflow-card">

        <div class="workflow-number">
            02 · PREPROCESSING
        </div>

        <div class="workflow-title">
            Standardization
        </div>

        <div class="workflow-text">
            Audio is converted to 16 kHz mono, normalized,
            and converted into a fixed-length representation.
        </div>

    </div>
    """)


with step3:

    st.html("""
    <div class="workflow-card">

        <div class="workflow-number">
            03 · FEATURES
        </div>

        <div class="workflow-title">
            Mel Spectrogram
        </div>

        <div class="workflow-text">
            STFT and Mel filtering transform the waveform
            into a 64-band log Mel spectrogram.
        </div>

    </div>
    """)


with step4:

    st.html("""
    <div class="workflow-card">

        <div class="workflow-number">
            04 · CLASSIFICATION
        </div>

        <div class="workflow-title">
            VaaniNet
        </div>

        <div class="workflow-text">
            The custom CNN analyzes the spectrogram and
            predicts REAL or FAKE with probabilities.
        </div>

    </div>
    """)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.html("""
<div class="section-title">
    VaaniNet V3 Performance
</div>

<div class="section-description">
    Results obtained on a held-out test set.
</div>
""")


metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.html("""
    <div class="prob-card">

        <div class="prob-label">
            Accuracy
        </div>

        <div class="prob-value">
            79.87%
        </div>

    </div>
    """)


with metric2:

    st.html("""
    <div class="prob-card">

        <div class="prob-label">
            F1 Score
        </div>

        <div class="prob-value">
            75.76%
        </div>

    </div>
    """)


with metric3:

    st.html("""
    <div class="prob-card">

        <div class="prob-label">
            ROC-AUC
        </div>

        <div class="prob-value">
            0.9738
        </div>

    </div>
    """)


st.html("""
<div class="model-card">

    <b>Model:</b> VaaniNet V3
    &nbsp; · &nbsp;
    <b>Architecture:</b> Custom CNN
    &nbsp; · &nbsp;
    <b>Training:</b> From scratch
    &nbsp; · &nbsp;
    <b>Parameters:</b> ~422K

</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    VaaniRakshak-AI · VaaniNet V3 · Voice Authenticity Detection

</div>
""")