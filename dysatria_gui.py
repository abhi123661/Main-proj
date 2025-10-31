import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import joblib
import threading
import time

# ---------------- Paths & Parameters ----------------
MODEL_PATH = "best_model_mfcc_delta.keras"
SCALER_PATH = "scaler_mfcc_delta.save"
dataset_sr = 16000
CNN_FEATURE_LEN = 32  # must match your trained CNN input

# ---------------- Load Model and Scaler ----------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- Audio Preprocessing ----------------
def preprocess_audio(y):
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y)
    y = librosa.effects.preemphasis(y)
    return y

def extract_features_live(y, n_mfcc=16):
    if len(y) == 0:
        return None
    y = preprocess_audio(y)
    mfcc = librosa.feature.mfcc(y=y, sr=dataset_sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, delta])

    # -------- Pad/Truncate to match CNN input --------
    if features.shape[1] < CNN_FEATURE_LEN:
        pad_width = CNN_FEATURE_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
    elif features.shape[1] > CNN_FEATURE_LEN:
        features = features[:, :CNN_FEATURE_LEN]

    features_mean = np.mean(features, axis=1)
    features_mean = np.nan_to_num(features_mean)

    # Scale features
    scaled = scaler.transform(features_mean.reshape(1, -1))
    # Reshape for CNN
    scaled = scaled.reshape(1, scaled.shape[1], 1, 1)
    return scaled

# ---------------- Recording ----------------
def record_audio(duration=8):
    """Record audio for live prediction."""
    recording = sd.rec(int(duration*dataset_sr), samplerate=dataset_sr, channels=1)
    for i in range(duration):
        progress_var.set((i+1)/duration*100)
        root.update_idletasks()
        time.sleep(1)
    sd.wait()
    y = recording.flatten()
    progress_var.set(0)
    return y

# ---------------- Prediction ----------------
def predict_file(file_path):
    y, _ = librosa.load(file_path, sr=dataset_sr)
    features = extract_features_live(y)
    if features is not None:
        pred = model.predict(features)[0][0]
        label = "Dysarthria" if pred>=0.5 else "Non-Dysarthria"
        messagebox.showinfo("Prediction", f"{label}\nProbability: {pred:.2f}")
    else:
        messagebox.showerror("Error", "Could not extract features.")

# ---------------- Multi-sentence Prediction ----------------
def multi_sentence_predict_thread():
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "I love eating fresh fruits every morning.",
        "Today is a beautiful day to learn something new."
    ]
    reps = 2
    probs = []

    for s in sentences:
        for r in range(reps):
            messagebox.showinfo("Read Sentence", f"Please read aloud:\n\n{s}\n(Repetition {r+1})")
            y = record_audio(duration=8)  # longer duration
            # split into overlapping 3-sec chunks
            chunk_size = dataset_sr * 3
            step_size = dataset_sr * 1  # 1 sec step
            chunk_probs = []
            for start in range(0, len(y), step_size):
                end = start + chunk_size
                chunk = y[start:end]
                if len(chunk) < chunk_size:
                    break  # ignore last too-short chunk
                features = extract_features_live(chunk)
                if features is not None:
                    pred = model.predict(features)[0][0]
                    chunk_probs.append(pred)
            if chunk_probs:
                probs.append(np.mean(chunk_probs))  # average over chunks

    if len(probs) == 0:
        messagebox.showerror("Error","No valid recordings detected")
        return

    avg = np.mean(probs)
    label = "Dysarthria" if avg>=0.5 else "Non-Dysarthria"
    messagebox.showinfo("Final Prediction", f"{label}\nAverage Probability: {avg:.2f}")

def multi_sentence_predict():
    threading.Thread(target=multi_sentence_predict_thread).start()

# ---------------- File Upload ----------------
def upload_file():
    path = filedialog.askopenfilename(filetypes=[("WAV files","*.wav")])
    if path:
        predict_file(path)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Dysarthria Detection")
root.geometry("450x350")

tk.Button(root,text="Upload WAV File",command=upload_file,width=30,height=2).pack(pady=10)
tk.Button(root,text="Read Sentences & Predict",command=multi_sentence_predict,width=30,height=2).pack(pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root,variable=progress_var,maximum=100)
progress_bar.pack(fill='x', padx=20, pady=20)

root.mainloop()
