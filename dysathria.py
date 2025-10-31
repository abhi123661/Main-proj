# test1.py - Corrected
import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
from tensorflow.keras.models import load_model
import parselmouth
from parselmouth.praat import call

SR = 16000
DEFAULT_RECORD_SECONDS = 12

# ==============================
# Load models
# ==============================
PARK_MODEL = "small_model.pkl"
PARK_SCALER = "small_scaler.pkl"
PARK_FEATURES = "small_features.pkl"

DYS_MODEL = "best_model_mfcc_delta.keras"
DYS_SCALER = "scaler_mfcc_delta.save"

clf = joblib.load(PARK_MODEL)
scaler_parkinson = joblib.load(PARK_SCALER)
features_parkinson = joblib.load(PARK_FEATURES)

dys_model = load_model(DYS_MODEL)
scaler_dys = joblib.load(DYS_SCALER)

# ==============================
# Audio utilities
# ==============================
def save_wav_array(path, y, sr=SR):
    max_abs = np.max(np.abs(y))
    if max_abs > 1.0:
        y = y / max_abs
    sf.write(path, y, sr, subtype='PCM_16')

def record_audio(duration=DEFAULT_RECORD_SECONDS):
    st.info(f"Recording {duration} seconds...")
    rec = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return rec.flatten()

def plot_spectrogram(y):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(8,3))
    librosa.display.specshow(D, sr=SR, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt

# ==============================
# Parkinson’s Feature Extraction (identical to training)
# ==============================
def build_feature_row(audio_path):
    base = {f: 0.0 for f in features_parkinson}
    try:
        snd = parselmouth.Sound(audio_path)
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # --- Jitter ---
        jitter_map = {
            "locPctJitter":"Get jitter (local)",
            "locAbsJitter":"Get jitter (local, absolute)",
            "rapJitter":"Get jitter (rap)",
            "ppq5Jitter":"Get jitter (ppq5)",
            "ddpJitter":"Get jitter (ddp)"
        }
        for k, v in jitter_map.items():
            if k in base:
                try:
                    base[k] = float(call(pp, v, 0, 0, 75, 500, 1.3))
                except:
                    base[k] = 0.0

        # --- Shimmer ---
        shimmer_map = {
            "locShimmer":"Get shimmer (local)",
            "locDbShimmer":"Get shimmer (local_dB)",
            "apq3Shimmer":"Get shimmer (apq3)",
            "apq5Shimmer":"Get shimmer (apq5)",
            "apq11Shimmer":"Get shimmer (apq11)",
            "ddaShimmer":"Get shimmer (dda)"
        }
        for k, v in shimmer_map.items():
            if k in base:
                try:
                    base[k] = float(call([snd, pp], v, 0, 0, 75, 500, 1.3, 1.6))
                except:
                    base[k] = 0.0

        # --- HNR ---
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        if "meanAutoCorrHarmonicity" in base:
            try:
                base["meanAutoCorrHarmonicity"] = float(call(harm, "Get mean", 0, 0))
            except:
                base["meanAutoCorrHarmonicity"] = 0.0

        # --- Intensity / Log energy ---
        intensity = call(snd, "To Intensity", 75, 0)
        if "mean_Log_energy" in base:
            try:
                base["mean_Log_energy"] = float(call(intensity, "Get mean", 0, 0, "energy"))
            except:
                base["mean_Log_energy"] = 0.0
        if "std_Log_energy" in base:
            try:
                base["std_Log_energy"] = float(np.std(intensity.values.T))
            except:
                base["std_Log_energy"] = 0.0
        if "mean_delta_log_energy" in base:
            try:
                delta_energy = np.diff(intensity.values.T[0])
                base["mean_delta_log_energy"] = float(np.mean(delta_energy))
            except:
                base["mean_delta_log_energy"] = 0.0

        # --- MFCCs (mean, std, delta) ---
        y_audio, sr_file = librosa.load(audio_path, sr=SR, mono=True)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr_file, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds = np.std(mfcc, axis=1)
        mfcc_deltas = librosa.feature.delta(mfcc)
        mfcc_deltas_mean = np.mean(mfcc_deltas, axis=1)

        for i in range(13):
            mn = f"mean_MFCC_{i}th_coef"
            std = f"std_MFCC_{i}th_coef"
            delta = f"mean_{i}th_delta"
            if mn in base: base[mn] = float(mfcc_means[i])
            if std in base: base[std] = float(mfcc_stds[i])
            if delta in base: base[delta] = float(mfcc_deltas_mean[i])

    except Exception as e:
        print("Parkinson feature extraction error:", e)

    return pd.DataFrame([base], columns=features_parkinson)

def predict_parkinson(audio_path):
    row = build_feature_row(audio_path)
    Xs = scaler_parkinson.transform(row)
    y_pred = clf.predict(Xs)[0]
    try:
        proba = clf.predict_proba(Xs)[0][1]
    except:
        proba = None
    label = "Parkinson's" if y_pred == 1 else "Healthy"
    return label, proba

# ==============================
# Dysarthria functions
# ==============================
def extract_features_dys(audio_path, n_mfcc=16):
    y, sr_file = librosa.load(audio_path, sr=SR)
    y = librosa.effects.preemphasis(y)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_file, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta])
    return np.mean(feat, axis=1)

def predict_dysarthria(audio_path):
    feat = extract_features_dys(audio_path)
    feat_scaled = scaler_dys.transform([feat])
    feat_scaled = feat_scaled.reshape(1, feat_scaled.shape[1], 1, 1)
    pred = dys_model.predict(feat_scaled)[0][0]
    label = "Dysarthria" if pred >= 0.5 else "Non-Dysarthria"
    return label, float(pred)

# ==============================
# Streamlit UI
# ==============================
st.title("Neurological Speech Disorder Detection")
SENTENCES = ["The quick brown fox jumps over the lazy dog.",
             "She sells seashells by the seashore.",
             "Reading aloud helps detect speech issues."]
sentence_to_read = np.random.choice(SENTENCES)
st.markdown(f"**Please read aloud:** `{sentence_to_read}`")

option = st.selectbox("Audio source", ["Record", "Upload WAV"])
duration = st.number_input("Record duration (s)", 5, 60, 12)
uploaded_file = None
if option == "Upload WAV":
    uploaded_file = st.file_uploader("Upload WAV", type=["wav"])
record_button = st.button("Run Detection")
result_placeholder = st.empty()
spectro_placeholder = st.empty()

def process_audio(y):
    # Spectrogram
    try:
        plt_fig = plot_spectrogram(y)
        spectro_placeholder.pyplot(plt_fig)
        plt.close("all")
    except:
        pass

    # Save temporary WAV for feature extraction
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    save_wav_array(tmp_path, y)

    try:
        park_label, park_prob = predict_parkinson(tmp_path)
        dys_label, dys_prob = predict_dysarthria(tmp_path)
        msg = f"🧠 Parkinson's: {park_label}"
        if park_prob is not None:
            msg += f" (Conf: {park_prob:.2f})"
        msg += f"\n🗣 Dysarthria: {dys_label} (Prob: {dys_prob:.2f})"
        result_placeholder.code(msg)
    except Exception as e:
        result_placeholder.error(f"Prediction failed: {e}")

    try:
        os.remove(tmp_path)
    except:
        pass

if record_button:
    if option == "Record":
        y = record_audio(duration)
        process_audio(y)
    else:
        if uploaded_file is None:
            st.warning("Upload WAV file")
        else:
            uploaded_file.seek(0)
            data, sr_file = sf.read(io.BytesIO(uploaded_file.read()))
            if sr_file != SR:
                data = librosa.resample(np.mean(data, axis=1) if data.ndim>1 else data, sr_file, SR)
            else:
                data = np.mean(data, axis=1) if data.ndim>1 else data
            process_audio(data.astype(np.float32))
