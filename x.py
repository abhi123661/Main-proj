import os
import numpy as np
import pandas as pd
import joblib
import librosa
import sounddevice as sd
import wavio
import parselmouth
from parselmouth.praat import call
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import time
import threading
import random

# ------------------------- Config
MODEL_FILE = "small_model.pkl"
SCALER_FILE = "small_scaler.pkl"
FEATURES_FILE = "small_features.pkl"

# ------------------------- Load artifacts
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(FEATURES_FILE):
    raise RuntimeError("Missing model/scaler/features files. Run train_small_features.py first.")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_list = joblib.load(FEATURES_FILE)

# ------------------------- Calibration factor
calibration_factor = 1.0

def calibrate_microphone(duration=3, sr=16000):
    global calibration_factor
    messagebox.showinfo("Calibration", f"Please remain silent for {duration}s...")
    rec = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    rms_silence = np.sqrt(np.mean(rec**2))
    if rms_silence > 0:
        calibration_factor = 0.05 / rms_silence  # target RMS ~0.05
    messagebox.showinfo("Calibration", f"Calibration done. Factor: {calibration_factor:.3f}")

# ------------------------- Sentence list
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Pack my box with five dozen water jugs.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "I am reading this sentence clearly for Parkinson's detection."
]

# ------------------------- GUI Setup
root = tk.Tk()
root.title("Parkinson's Detection (Multi-Sentence Real-time + Upload)")

sentence_label = tk.Label(root, text="", font=("Arial",12), wraplength=400)
sentence_label.pack(pady=5)
timer_label = tk.Label(root, text="", font=("Arial",10))
timer_label.pack(pady=2)
progress_label = tk.Label(root, text="", font=("Arial",10))
progress_label.pack(pady=2)
progress_bar = ttk.Progressbar(root, length=300, mode='determinate')
progress_bar.pack(pady=5)

# ------------------------- Feature Extraction Helpers
def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except:
        return default

def build_feature_row_uploaded(audio_path):
    base = {f:0.0 for f in feature_list}
    try:
        snd = parselmouth.Sound(audio_path)
        pp = call(snd, "To PointProcess (periodic, cc)", 75,500)
        # jitter
        jitter_map = {
            "locPctJitter":"Get jitter (local)",
            "locAbsJitter":"Get jitter (local, absolute)",
            "rapJitter":"Get jitter (rap)",
            "ppq5Jitter":"Get jitter (ppq5)",
            "ddpJitter":"Get jitter (ddp)"
        }
        for k,v in jitter_map.items():
            if k in base:
                try: base[k] = float(call(pp, v,0,0,75,500,1.3))
                except: base[k]=0.0
        # shimmer
        shimmer_map = {
            "locShimmer":"Get shimmer (local)",
            "locDbShimmer":"Get shimmer (local_dB)",
            "apq3Shimmer":"Get shimmer (apq3)",
            "apq5Shimmer":"Get shimmer (apq5)",
            "apq11Shimmer":"Get shimmer (apq11)",
            "ddaShimmer":"Get shimmer (dda)"
        }
        for k,v in shimmer_map.items():
            if k in base:
                try: base[k] = float(call([snd, pp], v,0,0,75,500,1.3,1.6))
                except: base[k]=0.0
        # HNR
        harm = call(snd,"To Harmonicity (cc)",0.01,75,0.1,1.0)
        if "meanAutoCorrHarmonicity" in base:
            try: base["meanAutoCorrHarmonicity"]=float(call(harm,"Get mean",0,0))
            except: base["meanAutoCorrHarmonicity"]=0.0
        # intensity
        intensity = call(snd,"To Intensity",75,0)
        if "mean_Log_energy" in base:
            try: base["mean_Log_energy"]=float(call(intensity,"Get mean",0,0,"energy"))
            except: base["mean_Log_energy"]=0.0
        # Librosa MFCC + log-energy
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y *= calibration_factor  # Apply calibration
        if y.size>0:
            if np.mean(np.abs(y)) < 0.01:  # too silent
                return None
            mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=2048,hop_length=512)
            mfcc_means = np.mean(mfcc,axis=1)
            mfcc_stds = np.std(mfcc,axis=1)
            mfcc_d1 = librosa.feature.delta(mfcc)
            mfcc_d1_means = np.mean(mfcc_d1,axis=1)
            for i in range(13):
                mn = f"mean_MFCC_{i}th_coef"
                std = f"std_MFCC_{i}th_coef"
                delta = f"mean_{i}th_delta"
                if mn in base: base[mn]=float(mfcc_means[i])
                if std in base: base[std]=float(mfcc_stds[i])
                if delta in base: base[delta]=float(mfcc_d1_means[i])
            # log-energy std & delta
            S = np.abs(librosa.stft(y,n_fft=2048,hop_length=512))
            frame_energy = np.sum(S**2,axis=0)+1e-10
            log_energy = np.log(frame_energy)
            if "std_Log_energy" in base: base["std_Log_energy"]=float(np.std(log_energy))
            if "mean_delta_log_energy" in base:
                d_log = np.diff(log_energy) if log_energy.size>1 else np.array([0])
                base["mean_delta_log_energy"]=float(np.mean(d_log))
    except: pass
    return pd.DataFrame([base],columns=feature_list)

def predict_dataframe(df_row):
    Xs = scaler.transform(df_row)
    y = model.predict(Xs)[0]
    try:
        proba = model.predict_proba(Xs)[0][1]  # Parkinson's probability specifically
    except:
        proba = None
    return int(y), proba

# ------------------------- GUI Prediction Functions
def predict_from_file(path):
    try:
        row = build_feature_row_uploaded(path)
        if row is None:
            messagebox.showerror("Error","Audio too silent or invalid.")
            return
        y,p = predict_dataframe(row)
        label = "Parkinson's" if y==1 else "Healthy"
        if p is not None: messagebox.showinfo("Result", f"{label}\nConfidence: {p:.2f}")
        else: messagebox.showinfo("Result", label)
    except Exception as e:
        messagebox.showerror("Error",f"Prediction failed:\n{e}")

def upload_and_predict():
    fname = filedialog.askopenfilename(filetypes=[("WAV files","*.wav")])
    if fname: predict_from_file(fname)

# ------------------------- Real-time Multi-Sentence Prediction with Progress Bar
def record_multiple_sentences(num_sentences=3, seconds=5, chunk_sec=3, threshold=0.6, sr=16000):
    sentences_to_read = random.sample(sentences, num_sentences)
    avg_probs = []
    progress_bar['maximum'] = num_sentences
    progress_bar['value'] = 0

    def read_sentence(idx):
        sentence_label.config(text=f"Sentence {idx+1}/{num_sentences}: {sentences_to_read[idx]}")
        for t in range(3,0,-1):
            timer_label.config(text=f"Recording starts in {t}s")
            root.update()
            time.sleep(1)
        timer_label.config(text="Recording now...")
        rec = sd.rec(int(seconds*sr),samplerate=sr,channels=1,dtype='float32')
        sd.wait()
        # Split into chunks
        num_chunks = int(np.ceil(seconds / chunk_sec))
        for c in range(num_chunks):
            start = int(c*chunk_sec*sr)
            end = int(min((c+1)*chunk_sec*sr, rec.shape[0]))
            chunk = rec[start:end,0]
            fname = f"temp_chunk_{idx}_{c}.wav"
            wavio.write(fname, chunk, sr, sampwidth=2)
            row = build_feature_row_uploaded(fname)
            try: os.remove(fname)
            except: pass
            if row is None:
                continue
            _, p = predict_dataframe(row)
            if p is not None:
                avg_probs.append(p)
        progress_bar['value'] += 1
        root.update()

    def run_all():
        for i in range(num_sentences):
            progress_label.config(text=f"Progress: {i}/{num_sentences}")
            read_sentence(i)
        if not avg_probs:
            messagebox.showerror("Error","No valid audio detected.")
            return
        final_prob = np.mean(avg_probs)
        final_label = 1 if final_prob > threshold else 0
        label_text = "Parkinson's" if final_label==1 else "Healthy"
        messagebox.showinfo("Final Result", f"{label_text}\nConfidence: {final_prob:.2f}")

        sentence_label.config(text="")
        progress_label.config(text="")
        timer_label.config(text="")
        progress_bar['value'] = 0

    threading.Thread(target=run_all).start()

# ------------------------- GUI Buttons
tk.Label(root, text="🎤 Parkinson's Detection", font=("Arial",14)).pack(pady=10)
tk.Button(root, text="Calibrate Microphone", command=lambda: calibrate_microphone(3), font=("Arial",12)).pack(pady=6)
tk.Button(root, text="Record Multiple Sentences & Predict", command=lambda: record_multiple_sentences(3,8,3), font=("Arial",12)).pack(pady=6)
tk.Button(root, text="Upload WAV & Predict", command=upload_and_predict, font=("Arial",12)).pack(pady=6)
tk.Label(root, text="Read sentences clearly for accurate detection.\nModel trained on robust MFCC + Praat features.", font=("Arial",9)).pack(pady=8)

root.mainloop()
