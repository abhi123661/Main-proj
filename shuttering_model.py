import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import wavio
import librosa
import numpy as np
import noisereduce as nr
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
import os

# ---------------- Load Trained Model ----------------
model_path = "best_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")
model = tf.keras.models.load_model(model_path)

# ---------------- Feature Extraction ----------------
def extract_features(audio_data, sr):
    # Extract 16 MFCCs and take mean
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=16)
    mfccs_mean = np.mean(mfccs, axis=1)
    # Apply min-max scaling as done during training
    mfccs_mean = minmax_scale(mfccs_mean, axis=0)
    # Reshape for CNN input
    return mfccs_mean.reshape(1, 16, 1, 1)

def preprocess_audio(file_path):
    # Load audio at 16 kHz
    audio_data, sr = librosa.load(file_path, sr=16000)
    # Noise reduction
    audio_data = nr.reduce_noise(y=audio_data, sr=sr)
    # Extract features
    features = extract_features(audio_data, sr)
    return features

def predict_audio(file_path):
    try:
        features = preprocess_audio(file_path)
        pred_prob = model.predict(features, verbose=0)[0][0]
        label = "Dysarthria" if pred_prob >= 0.5 else "Non-Dysarthria"
        messagebox.showinfo("Prediction", f"Prediction: {label}\nProbability: {pred_prob:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process audio: {str(e)}")

# ---------------- Record Real-Time Audio ----------------
def record_audio(duration=10):
    filename = "recorded.wav"
    fs = 16000
    messagebox.showinfo("Recording", f"Recording for {duration} seconds. Please read the sentence aloud.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    predict_audio(filename)

# ---------------- Upload WAV File ----------------
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        predict_audio(file_path)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Speech Disorder Detection")
root.geometry("500x400")

tk.Label(root, text="Speech Disorder Detection", font=("Helvetica", 16, "bold")).pack(pady=10)
tk.Label(root, text="Choose an option:", font=("Helvetica", 12)).pack(pady=5)

tk.Button(root, text="Upload WAV File", command=upload_file, width=35, height=2).pack(pady=10)
tk.Button(root, text="Record Real-Time Voice", command=record_audio, width=35, height=2).pack(pady=10)

tk.Label(root, text="Please read the following sentence aloud:", font=("Helvetica", 12)).pack(pady=10)
tk.Label(root, text='"The quick brown fox jumps over the lazy dog."', wraplength=450, font=("Helvetica", 11)).pack(pady=5)

root.mainloop()
