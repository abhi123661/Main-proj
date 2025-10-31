#!/usr/bin/env python3
"""
Streamlit single-file app (completed):
- asks user to select language
- shows sentences in selected language
- when Start pressed: uses OpenCV local window for webcam + dlib lip landmarks,
  uses sounddevice for audio capture, prompts user (TTS), records 8s per sentence,
  runs rule-based stutter detection (repetition/prolongation/block),
  runs Parkinson & Dysarthria models if present, shows results in Streamlit and
  also shows a final OpenCV result window (same style as local OpenCV script).
"""
import os
import time
import threading
import traceback
from collections import deque
import queue as pyqueue

import numpy as np
import cv2
import dlib
import sounddevice as sd
import librosa
import joblib
from imutils import face_utils

import streamlit as st

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

# Optional TF (Dysarthria)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------- CONFIG (edit if needed) ----------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

PD_MODEL_PATH = "parkinsons_rf_model.pkl"
PD_SCALER_PATH = "parkinsons_rf_scaler.pkl"
PD_FEATURES_PATH = "parkinsons_features.pkl"

DYS_MODEL_PATH = "best_model_mfcc_delta.keras"
DYS_SCALER_PATH = "scaler_mfcc_delta.save"

CAM_INDEX = 0
SR = 16000
RECORD_SECONDS_PER_SENTENCE = 8   # 8s per sentence
FPS_TARGET = 25

MIN_VOICED_SECONDS = 1.5
TOP_DB_TRIM = 35

# Sentences per language (5 each). You can replace these with better translations.
SENTENCES_MAP = {
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "I am reading this sentence aloud to test speech.",
        "Today is a sunny day and we enjoy the weather.",
        "Artificial intelligence can help detect speech disorders."
    ],
    "hindi": [
        "तेज़ भूरी लोमड़ी आलसी कुत्ते पर छलाँग लगाती है।",
        "वह समुद्रतट पर सीप बेचती है।",
        "मैं इस वाक्य को ज़ोर से पढ़ रहा/रही हूँ।",
        "आज धूप वाला दिन है और हम मौसम का आनंद लेते हैं।",
        "कृत्रिम बुद्धिमत्ता भाषण विकारों का पता लगाने में मदद कर सकती है।"
    ],
    "kannada": [
        "ದ್ರುತ ಕಣಿವೆ ಸೋರಣ ಕುತೂಹಲದ ನಾಯಿಯ ಮೇಲೆ ಜಿಗಿಯುತ್ತದೆ.",
        "ಅವಳು ಸಮುದ್ರತೀರದ ಬಳಿ ಶೆಲ್‌ಗಳನ್ನು ಮಾರುತ್ತಾಳೆ.",
        "ನಾನು ಈ ವಾಕ್ಯವನ್ನು ಓದೇನು ಪರೀಕ್ಷೆಗಾಗಿ.",
        "ಇಂದು ಸೂರ್ಯನ ದಿನ, ನಾವು ಹವಾಮಾನವನ್ನು ಆನಂದಿಸುತ್ತೇವೆ.",
        "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಭಾಷಣದ ಬಾಧೆಗಳನ್ನು ಕಂಡುಹಿಡಿಯಲು ಸಹಾಯ ಮಾಡಬಹುದು."
    ],
    "telugu": [
        "ద్రుత బూడిద కోతి ఆలస్యమైన కుక్కపై దూకుతుంది.",
        "ఆమె తీరానికి అటవీ షెల్‌లు అమ్ముతుంది.",
        "నేను ఈ వాక్యాన్ని పైసగానే చదువుతున్నాను.",
        "నేడు సూర్యుడు ప్రకాశవంతమైన రోజు, మనం వాతావరణాన్ని ఆస్వాదిస్తున్నాము.",
        "కృతక మేధస్సు మాట్లాడే సమస్యలను గుర్తించడంలో సహాయపడుతుంది."
    ],
    "tamil": [
        "விரைவான கருப்பு நரி சோம்பேறி நாயின் மேலே குதிக்கிறது.",
        "அவள் கடற்கரையில் சிப்பெல்களை விற்பனை செய்கிறாள்.",
        "நான் இந்த வாக்கியத்தை சோதனைக்காக படிக்கிறேன்.",
        "இன்று ஒரு சூரிய ஒளி நாள்; நாம் வானிலை அனுபவிக்கிறோம்.",
        "செயற்கை நுண்ணறிவு பேசும் குறைபாடுகளை கண்டறிய உதவும்."
    ]
}

LANG_DISPLAY = {
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "telugu": "Telugu",
    "tamil": "Tamil"
}

LANG_KEYS = list(LANG_DISPLAY.keys())

# language-specific thresholds for stutter rules (tunable)
LANG_THRESH = {
    "english":  {"block_silence": 0.7, "prolong_ms": 500, "rep_max_voiced": 0.18, "rep_min_repeats": 2},
    "hindi":    {"block_silence": 0.7, "prolong_ms": 550, "rep_max_voiced": 0.20, "rep_min_repeats": 2},
    "kannada":  {"block_silence": 0.7, "prolong_ms": 550, "rep_max_voiced": 0.20, "rep_min_repeats": 2},
    "telugu":   {"block_silence": 0.7, "prolong_ms": 550, "rep_max_voiced": 0.20, "rep_min_repeats": 2},
    "tamil":    {"block_silence": 0.7, "prolong_ms": 550, "rep_max_voiced": 0.20, "rep_min_repeats": 2},
}

# ---------------- TTS init ----------------
def _init_tts():
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        return engine
    except Exception:
        return None

TTS_ENGINE = _init_tts()

def speak(text):
    if TTS_ENGINE:
        # run in background to avoid blocking
        threading.Thread(target=lambda: (TTS_ENGINE.say(text), TTS_ENGINE.runAndWait()), daemon=True).start()
    else:
        # fallback: streamlit write / print
        print("[TTS]", text)

# ----------------- Model availability & loading ----------------
PD_AVAILABLE = os.path.exists(PD_MODEL_PATH) and os.path.exists(PD_SCALER_PATH) and os.path.exists(PD_FEATURES_PATH)
DYS_AVAILABLE = os.path.exists(DYS_MODEL_PATH) and os.path.exists(DYS_SCALER_PATH) and TF_AVAILABLE

PD_MODEL = PD_SCALER = PD_FEATURES = None
DYS_MODEL = DYS_SCALER = None

if PD_AVAILABLE:
    try:
        PD_MODEL = joblib.load(PD_MODEL_PATH)
        PD_SCALER = joblib.load(PD_SCALER_PATH)
        PD_FEATURES = joblib.load(PD_FEATURES_PATH)
        print("Loaded Parkinson artifacts.")
    except Exception as e:
        print("Failed to load Parkinson model artifacts:", e)
        PD_AVAILABLE = False

if DYS_AVAILABLE:
    try:
        DYS_MODEL = load_model(DYS_MODEL_PATH)
        DYS_SCALER = joblib.load(DYS_SCALER_PATH)
        print("Loaded Dysarthria artifacts.")
    except Exception as e:
        print("Failed to load Dysarthria artifacts:", e)
        DYS_AVAILABLE = False

# ---------------- dlib predictor ----------------
if not os.path.exists(PREDICTOR_PATH):
    print(f"WARNING: Missing {PREDICTOR_PATH}. Lip landmarks disabled.")
    predictor = None
else:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

detector = dlib.get_frontal_face_detector()

# ---------------- audio capture (global) ----------------
AUDIO_BUFFER = deque()        # will store float samples
AUDIO_RMS = deque(maxlen=500)
_audio_lock = threading.Lock()
_audio_stream = None

def audio_callback(indata, frames, time_info, status):
    try:
        mono = indata[:, 0] if indata.ndim == 2 else indata
    except Exception:
        mono = indata.flatten()
    with _audio_lock:
        AUDIO_BUFFER.extend(mono.tolist())
        if mono.size > 0:
            rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)) + 1e-12)
            AUDIO_RMS.append(rms)

def start_audio_stream():
    global _audio_stream
    if _audio_stream is None:
        _audio_stream = sd.InputStream(channels=1, samplerate=SR, blocksize=1024, callback=audio_callback)
        _audio_stream.start()
        print("Audio stream started.")

def stop_audio_stream():
    global _audio_stream
    if _audio_stream is not None:
        try:
            _audio_stream.stop()
            _audio_stream.close()
        except Exception:
            pass
        _audio_stream = None
        print("Audio stream stopped.")

# ---------------- feature helpers (same logic) ----------------
def compute_pd_features(y, sr):
    feats = {}
    y = y.astype(np.float32)
    if y.size == 0:
        return feats
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    for i in range(13):
        feats[f"mean_MFCC_{i}th_coef"] = float(mfcc_mean[i])
        feats[f"std_MFCC_{i}th_coef"] = float(mfcc_std[i])
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    for i in range(13):
        feats[f"mean_{i}th_delta"] = float(mfcc_delta_mean[i])
    hop = 512
    frame_len = 1024
    if len(y) < frame_len:
        y_pad = np.pad(y, (0, frame_len - len(y)), mode="constant")
    else:
        y_pad = y
    frames = librosa.util.frame(y_pad, frame_length=frame_len, hop_length=hop)
    energies = np.mean(frames ** 2, axis=0) + 1e-10
    log_energy = np.log(energies)
    feats["mean_Log_energy"] = float(np.mean(log_energy))
    feats["std_Log_energy"] = float(np.std(log_energy))
    d_log_energy = np.diff(log_energy)
    feats["mean_delta_log_energy"] = float(np.mean(d_log_energy)) if len(d_log_energy) > 0 else 0.0
    feats["locPctJitter"] = 0.0
    return feats

def vectorize_pd(feat_dict, ordered_feature_names, scaler):
    vec = [float(feat_dict.get(name, 0.0)) for name in ordered_feature_names]
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    return scaler.transform(vec)

def extract_dys_features(y, sr, n_mfcc=16):
    y_pre = librosa.effects.preemphasis(y)
    try:
        y_pre, _ = librosa.effects.trim(y_pre, top_db=TOP_DB_TRIM)
    except Exception:
        pass
    mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    features = np.concatenate([np.mean(mfcc, axis=1), np.mean(delta, axis=1)]).astype(np.float32)
    return features

def vectorize_dys(feat_vec, scaler):
    X = scaler.transform(feat_vec.reshape(1, -1))
    X = X.reshape(-1, X.shape[1], 1, 1)
    return X

# ---------------- stutter rule-based primitives ----------------
def voiced_unvoiced_regions(y, sr, energy_thresh_ratio=0.02):
    if y.size == 0:
        return []
    hop = 256
    frame_len = 512
    # ensure length for framing
    if len(y) < frame_len:
        frames = np.array([y])
        energies = np.array([np.mean(y**2)])
    else:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).astype(np.float32)
        energies = np.mean(frames**2, axis=0)
    thr = max(np.median(energies) * energy_thresh_ratio, 1e-8)
    voiced = energies > thr
    segs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            s = start * hop / sr
            e = (i * hop + frame_len) / sr
            segs.append((max(0.0, s), min(e, len(y)/sr)))
            start = None
    if start is not None:
        s = start * hop / sr
        e = len(y)/sr
        segs.append((s, e))
    return segs

def spectral_flux(y, sr):
    if len(y) < 512:
        return 0.0
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    return np.mean(flux) if flux.size else 0.0

def detect_repetition_prolongation_block(y, sr, lang):
    res = {"repetition": False, "repetition_detail": None,
           "prolongation": False, "prolongation_detail": None,
           "block": False, "block_detail": None}
    cfg = LANG_THRESH.get(lang, LANG_THRESH["english"])
    block_thresh = cfg["block_silence"]
    prolong_ms = cfg["prolong_ms"]
    rep_max_voiced = cfg["rep_max_voiced"]
    rep_min_repeats = cfg["rep_min_repeats"]

    if y.size == 0:
        return res

    y_abs = np.abs(y)
    if np.max(y_abs) > 0:
        y_n = y / np.max(y_abs)
    else:
        y_n = y

    segs = voiced_unvoiced_regions(y_n, sr, energy_thresh_ratio=0.02)
    if len(segs) == 0:
        if len(y)/sr > block_thresh:
            res["block"] = True
            res["block_detail"] = {"reason": "all_silent", "duration": len(y)/sr}
        return res

    # gaps between voiced segs
    gaps = []
    last_end = 0.0
    for (s,e) in segs:
        gap = s - last_end
        if gap > 0:
            gaps.append(gap)
        last_end = e
    final_gap = max(0.0, len(y)/sr - segs[-1][1])
    if final_gap > 0:
        gaps.append(final_gap)
    long_gaps = [g for g in gaps if g >= block_thresh]
    if long_gaps:
        res["block"] = True
        res["block_detail"] = {"long_gaps": long_gaps}

    # repetition detection: repeated short voiced segments
    short_voiced = [ (s,e) for (s,e) in segs if (e-s) <= rep_max_voiced ]
    if len(short_voiced) >= rep_min_repeats:
        # indices
        indices = [i for i,(s,e) in enumerate(segs) if (e-s) <= rep_max_voiced]
        consec_runs = 0
        max_run = 0
        prev_idx = -10
        for idx in indices:
            if idx == prev_idx + 1:
                consec_runs += 1
            else:
                consec_runs = 1
            prev_idx = idx
            if consec_runs > max_run:
                max_run = consec_runs
        if max_run >= rep_min_repeats:
            res["repetition"] = True
            durations = [e-s for (s,e) in short_voiced]
            res["repetition_detail"] = {"short_count": len(short_voiced), "max_consec": max_run, "durations": durations}

    # prolongation detection: voiced segment longer than threshold AND low spectral flux
    for (s,e) in segs:
        dur = (e-s)
        if dur * 1000.0 >= prolong_ms:
            i0 = int(max(0, int(s*sr)))
            i1 = int(min(len(y), int(e*sr)))
            chunk = y_n[i0:i1] if i1 > i0 else np.array([])
            if chunk.size == 0:
                continue
            flux = spectral_flux(chunk, sr)
            if flux < 0.5:
                res["prolongation"] = True
                res["prolongation_detail"] = {"start": s, "end": e, "dur_s": dur, "flux": float(flux)}
                break

    return res

# ---------------- detection worker (runs in background thread) ----------------
def run_detection_job(lang, sentences, status_callback, stop_event):
    """
    status_callback(tag, text_or_obj) - called to enqueue messages back to Streamlit UI.
    The OpenCV window will show live face+lip dots + countdown per sentence.
    stop_event: threading.Event used to request stop from main thread (safe).
    """
    status_callback("info", translations[lang]["starting_record"])
    try:
        # start audio stream
        start_audio_stream()
    except Exception as e:
        status_callback("error", f"Audio stream error: {e}")
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        status_callback("error", translations[lang]["camera_fail"])
        stop_audio_stream()
        return

    lip_history = []
    audio_history = []
    audio_levels_history = []

    try:
        for idx, sent in enumerate(sentences):
            # check stop request before starting sentence
            if stop_event.is_set():
                status_callback("info", translations[lang]["stopped_by_user"])
                break

            prompt_text = f"{translations[lang]['sentence_prefix']} {idx+1}/{len(sentences)}: {sent}"
            status_callback("sentence", prompt_text)
            # always call speak with prompt_text
            speak(prompt_text)

            local_lip = []
            local_audio = []
            local_aud_levels = []

            t_end = time.time() + RECORD_SECONDS_PER_SENTENCE
            while time.time() < t_end:
                # allow stop mid-sentence as well (will finish current frame loop then break)
                if stop_event.is_set():
                    status_callback("info", translations[lang]["stopped_by_user"])
                    break

                ret, frame = cap.read()
                if not ret:
                    status_callback("error", translations[lang]["frame_fail"])
                    break
                display = frame.copy()

                if predictor is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray, 0)
                        for face in faces[:1]:
                            shape = predictor(gray, face)
                            shape = face_utils.shape_to_np(shape)
                            top_lip = np.mean(shape[50:53], axis=0)
                            bottom_lip = np.mean(shape[65:68], axis=0)
                            lip_dist = float(np.linalg.norm(top_lip - bottom_lip))
                            local_lip.append(lip_dist)

                            lips = shape[48:68].astype(int)
                            cv2.polylines(display, [lips], isClosed=True, color=(0, 0, 255), thickness=2)
                            for (x, y) in lips:
                                cv2.circle(display, (x, y), 1, (255, 0, 0), -1)
                            cv2.putText(display, f"LipDist: {lip_dist:.2f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    except Exception:
                        # predictor call failed for some reason; continue without lips
                        cv2.putText(display, "Predictor error - lip landmarks disabled", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                else:
                    cv2.putText(display, "Predictor missing - lip landmarks disabled", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                remaining = int(max(0, round(t_end - time.time())))
                cv2.putText(display, prompt_text, (10, display.shape[0]-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(display, f"Time left: {remaining}s", (10, display.shape[0]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                with _audio_lock:
                    if AUDIO_RMS:
                        median_rms = np.median(list(AUDIO_RMS))
                        local_aud_levels.append(median_rms)
                        cv2.putText(display, f"Mic RMS: {median_rms:.6f}", (display.shape[1]-280, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    if AUDIO_BUFFER:
                        n = min(len(AUDIO_BUFFER), int(0.2 * SR))
                        for _ in range(n):
                            local_audio.append(AUDIO_BUFFER.popleft())

                cv2.imshow(translations[lang]["window_title"], display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    status_callback("info", translations[lang]["user_quit"])
                    stop_event.set()
                    break

            lip_history.extend(local_lip)
            audio_history.extend(local_audio)
            audio_levels_history.extend(local_aud_levels)
            status_callback("info", translations[lang]["finished_sentence"].format(idx+1))
            time.sleep(0.2)

        # post-processing
        status_callback("info", translations[lang]["processing"])
        y_total = np.asarray(audio_history, dtype=np.float32).flatten()
        if y_total.size > 0:
            max_abs = np.max(np.abs(y_total))
            if max_abs > 0:
                y_total = y_total / max_abs
            try:
                y_trimmed, _ = librosa.effects.trim(y_total, top_db=TOP_DB_TRIM)
            except Exception:
                y_trimmed = y_total
        else:
            y_trimmed = y_total

        # compute smoothed lip and audio arrays used by stutter rule (optional)
        lip_arr = np.array(lip_history, dtype=np.float32) if lip_history else np.array([], dtype=np.float32)
        aud_arr = np.array(audio_levels_history, dtype=np.float32) if audio_levels_history else np.array([], dtype=np.float32)
        if lip_arr.size >= 5:
            kernel = np.ones(5, dtype=np.float32)/5.0
            sm_lip = np.convolve(lip_arr, kernel, mode='valid')
        else:
            sm_lip = lip_arr
        if aud_arr.size >= 5:
            kernel = np.ones(5, dtype=np.float32)/5.0
            sm_aud = np.convolve(aud_arr, kernel, mode='valid')
        else:
            sm_aud = aud_arr

        # rule-based stutter (language-specific)
        stutter_det = detect_repetition_prolongation_block(y_trimmed, SR, lang)

        # Parkinson & Dys processing
        pd_label = pd_prob = dys_label = dys_prob = None
        voiced_len = len(y_trimmed) / float(SR) if SR > 0 else 0.0

        if voiced_len < MIN_VOICED_SECONDS:
            pd_label = translations[lang]["insufficient_audio"]
            dys_label = translations[lang]["insufficient_audio"]
        else:
            if PD_AVAILABLE:
                try:
                    pd_feats_dict = compute_pd_features(y_trimmed, SR)
                    for f in PD_FEATURES:
                        pd_feats_dict.setdefault(f, 0.0)
                    Xpd = vectorize_pd(pd_feats_dict, PD_FEATURES, PD_SCALER)
                    pd_prob = float(PD_MODEL.predict_proba(Xpd)[0][1])
                    pd_label = translations[lang]["detected"] if pd_prob > 0.5 else translations[lang]["not_detected"]
                except Exception as e:
                    pd_label = f"Error: {e}"

            if DYS_AVAILABLE:
                try:
                    chunk_probs = []
                    sentences_chunks = librosa.effects.split(y_trimmed, top_db=TOP_DB_TRIM)
                    for start, end in sentences_chunks:
                        y_chunk = y_trimmed[start:end]
                        if len(y_chunk) / float(SR) < 0.5:
                            continue
                        dys_vec = extract_dys_features(y_chunk, SR)
                        Xd = vectorize_dys(dys_vec, DYS_SCALER)
                        prob = float(DYS_MODEL.predict(Xd, verbose=0)[0][0])
                        chunk_probs.append(prob)
                    if chunk_probs:
                        avg_prob = float(np.mean(chunk_probs))
                        dys_prob = avg_prob
                        dys_label = translations[lang]["detected"] if avg_prob > 0.8 else translations[lang]["not_detected"]
                    else:
                        dys_label = translations[lang]["insufficient_audio"]
                except Exception as e:
                    dys_label = f"Error: {e}"

        final_result = {
            "Parkinson": pd_label,
            "Parkinson_prob": pd_prob,
            "Dysarthria": dys_label,
            "Dysarthria_prob": dys_prob,
            "Stutter_rule": stutter_det
        }

        # Also present a final OpenCV result image like the local script
        try:
            srpt = stutter_det
            lines = [
                f"Parkinson: {final_result['Parkinson']} | prob: {final_result['Parkinson_prob']}",
                f"Dysarthria: {final_result['Dysarthria']} | prob: {final_result['Dysarthria_prob']}",
                f"Stutter - Repetition: {srpt.get('repetition', False)} (shorts={srpt.get('repetition_detail',{}).get('short_count') if srpt.get('repetition_detail') else 'N/A'})",
                f"Stutter - Prolongation: {srpt.get('prolongation', False)}",
                f"Stutter - Block: {srpt.get('block', False)}",
                f"Speech voiced seconds: {voiced_len:.2f}",
            ]
            width = 1000
            height = 40 * len(lines) + 40
            result_img = 255 * np.ones((height, width, 3), dtype=np.uint8)
            y0 = 30
            for ln in lines:
                cv2.putText(result_img, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y0 += 40
            # Show result image window
            cv2.imshow("Final Results - Press any key to close", result_img)
            key = cv2.waitKey(0)  # Waits until a key is pressed
            cv2.destroyAllWindows()

        except Exception:
            pass

        status_callback("result", final_result)
    except Exception as e:
        status_callback("error", f"Exception during detection: {e}\n{traceback.format_exc()}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        stop_audio_stream()


# ---------------- translations / UI text per language ----------------
translations = {
    "english": {
        "lang_name": "English",
        "start_btn": "Start Detection",
        "stop_btn": "Stop",
        "sentence_prefix": "Please read",
        "starting_record": "Starting recording...",
        "camera_fail": "Cannot open camera (check permissions).",
        "frame_fail": "Frame not read from camera.",
        "processing": "Post-processing and running detectors...",
        "finished_sentence": "Finished sentence {}.",
        "insufficient_audio": "Insufficient audio",
        "detected": "Detected",
        "not_detected": "Not Detected",
        "window_title": "Recording - press 'q' to quit",
        "user_quit": "User quit recording.",
        "stopped_by_user": "Stopped by user request."
    },
    "hindi": {
        "lang_name": "हिन्दी",
        "start_btn": "रिकॉर्डिंग शुरू करें",
        "stop_btn": "रोकें",
        "sentence_prefix": "कृपया पढ़िए",
        "starting_record": "रिकॉर्डिंग शुरू हो रही है...",
        "camera_fail": "कैमरा नहीं खुल सका (अनुमतियाँ जाँचें)।",
        "frame_fail": "कैमरा फ्रेम नहीं पढ़ा गया।",
        "processing": "प्रोसेस कर रहा है...",
        "finished_sentence": "वाक्य {} समाप्त हुआ।",
        "insufficient_audio": "पर्याप्त ऑडियो नहीं",
        "detected": "मिला",
        "not_detected": "नहीं मिला",
        "window_title": "रिकॉर्डिंग - 'q' दबाएँ बंद करने के लिए",
        "user_quit": "उपयोगकर्ता ने रुकवा दिया।",
        "stopped_by_user": "उपयोगकर्ता के अनुरोध पर रोका गया।"
    },
    "kannada": {
        "lang_name": "Kannada",
        "start_btn": "ಶುರುಮಾಡಿ",
        "stop_btn": "ನಿಲ್ಲಿಸಿ",
        "sentence_prefix": "ದಯವಿಟ್ಟು ಓದಿ",
        "starting_record": "ರೆಕಾರ್ಡಿಂಗ್ ಆರಂಭಿಸುತ್ತಿದೆ...",
        "camera_fail": "ಕ್ಯಾಮೆರಾ ತೆರೆಯಲಾಗುತ್ತಿಲ್ಲ.",
        "frame_fail": "ಫ್ರೇಮ್ ಓದಲಾಗಲಿಲ್ಲ.",
        "processing": "ಪ್ರೋಸೆಸಿಂಗ್...",
        "finished_sentence": "ವಾಕ್ಯ {} ಮುಗಿಯಿತು.",
        "insufficient_audio": "ಪ್ರಯುಕ್ತ ಶಬ್ದವಿಲ್ಲ",
        "detected": "ಕಂಡುಬಂದಿದೆ",
        "not_detected": "ಕಂಡುಬಂದಿಲ್ಲ",
        "window_title": "Recording - press 'q' to quit",
        "user_quit": "User requested quit.",
        "stopped_by_user": "User requested stop."
    },
    "telugu": {
        "lang_name": "తెలుగు",
        "start_btn": "రికార్డింగ్ ప్రారంభించు",
        "stop_btn": "ఆపు",
        "sentence_prefix": "దయచేసి చదవండి",
        "starting_record": "రికార్డింగ్ ప్రారంభమవుతోంది...",
        "camera_fail": "క్యామరాను తెరవలేదు.",
        "frame_fail": "ఫ్రేమ్ చదవబడలేదు.",
        "processing": "పోస్ట్-ప్రాసెసింగ్...",
        "finished_sentence": "వాక్యం {} ముగిసింది.",
        "insufficient_audio": "ప్రయాప్త ఆడియో లేదు",
        "detected": "గుర్తించబడింది",
        "not_detected": "గుర్తించబడలేదు",
        "window_title": "Recording - press 'q' to quit",
        "user_quit": "User requested quit.",
        "stopped_by_user": "User requested stop."
    },
    "tamil": {
        "lang_name": "Tamil",
        "start_btn": "பதிவு தொடங்கு",
        "stop_btn": "நிறுத்து",
        "sentence_prefix": "தயவு செய்து வாசிக்கவும்",
        "starting_record": "பதிவு தொடங்குகிறது...",
        "camera_fail": "கேமரா திறக்க முடியவில்லை.",
        "frame_fail": "படம் வாசிக்கப்படவில்லை.",
        "processing": "போஸ்ட்-ப்ரோசெஸிங்...",
        "finished_sentence": "வாக்கியம் {} முடிந்தது.",
        "insufficient_audio": "போதுமான ஒலி இல்லை",
        "detected": "கண்டறியப்பட்டது",
        "not_detected": "கண்டறியப்படவில்லை",
        "window_title": "Recording - press 'q' to quit",
        "user_quit": "User requested quit.",
        "stopped_by_user": "User requested stop."
    }
}

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide", page_title="Speech Disorder Detection (Streamlit + OpenCV)")
st.title("Speech Disorder Detection — Streamlit + OpenCV (Multilingual)")

col1, col2 = st.columns([2,1])
with col1:
    lang_choice = st.selectbox("Select language / भाषा चुनें:", options=LANG_KEYS, format_func=lambda k: LANG_DISPLAY[k])
    sentences = SENTENCES_MAP[lang_choice]
    st.markdown("### Sentences (read these aloud when recording starts):")
    for i, s in enumerate(sentences, 1):
        st.write(f"{i}. {s}")

with col2:
    st.write("Controls")
    start_btn = st.button(translations[lang_choice]["start_btn"])
    stop_btn = st.button(translations[lang_choice]["stop_btn"])
    status_box = st.empty()
    result_box = st.empty()

# Shared job state
if "JOB_STATE" not in st.session_state:
    st.session_state["JOB_STATE"] = {
        "running": False,
        "thread": None,
        "queue": pyqueue.Queue(),
        "stop_event": threading.Event()
    }

# This function will be passed to the worker; it enqueues messages into JOB_STATE queue
def status_enqueue(tag, payload):
    try:
        st.session_state["JOB_STATE"]["queue"].put((tag, payload))
    except Exception:
        # if session_state not accessible for some reason, print fallback
        print("Status enqueue failed:", tag, payload)

# Start detection
if start_btn and not st.session_state["JOB_STATE"]["running"]:
    # reset/clear stop event and queue
    st.session_state["JOB_STATE"]["stop_event"].clear()
    with st.session_state["JOB_STATE"]["queue"].mutex:
        st.session_state["JOB_STATE"]["queue"].queue.clear()
    st.session_state["JOB_STATE"]["running"] = True
    worker = threading.Thread(target=run_detection_job,
                              args=(lang_choice, sentences, status_enqueue, st.session_state["JOB_STATE"]["stop_event"]),
                              daemon=True)
    st.session_state["JOB_STATE"]["thread"] = worker
    worker.start()
    status_box.info(translations[lang_choice]["starting_record"])

# Stop detection request (user asks to stop main job)
if stop_btn and st.session_state["JOB_STATE"]["running"]:
    # signal stop to worker via event
    st.session_state["JOB_STATE"]["stop_event"].set()
    status_enqueue("info", translations[lang_choice]["stopped_by_user"])
    st.session_state["JOB_STATE"]["running"] = False
    status_box.warning("Requested stop — worker will stop shortly (press 'q' inside camera window to abort immediately).")

# Drain messages from worker queue into a simple in-memory display queue and show them.
if "msgs" not in st.session_state:
    st.session_state["msgs"] = deque(maxlen=500)

# Move any queued messages (thread -> main) into session_state msgs
jq = st.session_state["JOB_STATE"]["queue"]
while not jq.empty():
    try:
        tag, payload = jq.get_nowait()
    except Exception:
        break
    st.session_state["msgs"].append((tag, payload))

# Display messages
while st.session_state["msgs"]:
    tag, payload = st.session_state["msgs"].popleft()
    if tag == "info":
        status_box.info(payload)
    elif tag == "error":
        status_box.error(payload)
    elif tag == "sentence":
        status_box.success(payload)
    elif tag == "result":
        # --- Nicely render results in Streamlit (human-friendly) ---
        result_box.subheader("Final Results")

        # top-level labels
        pd_label = payload.get("Parkinson", None)
        pd_prob = payload.get("Parkinson_prob", None)
        dys_label = payload.get("Dysarthria", None)
        dys_prob = payload.get("Dysarthria_prob", None)
        stutter = payload.get("Stutter_rule", {})

        with result_box:
            st.markdown("*Parkinson's:* " + (str(pd_label) if pd_label is not None else "N/A"))
            if pd_prob is not None:
                st.write(f"Probability: {pd_prob:.3f}")
            st.markdown("*Dysarthria:* " + (str(dys_label) if dys_label is not None else "N/A"))
            if dys_prob is not None:
                st.write(f"Probability: {dys_prob:.3f}")

            # Stutter summary
            st.markdown("### Stutter rule-based summary")
            st.write("Repetition detected:", stutter.get("repetition", False))
            if stutter.get("repetition_detail"):
                rd = stutter["repetition_detail"]
                st.write(f"- short_count: {rd.get('short_count')}, max_consec: {rd.get('max_consec')}")
                if "durations" in rd:
                    st.write(f"- durations (s): {', '.join([f'{d:.3f}' for d in rd.get('durations', [])])}")

            st.write("Prolongation detected:", stutter.get("prolongation", False))
            if stutter.get("prolongation_detail"):
                pdt = stutter["prolongation_detail"]
                st.write(f"- start: {pdt.get('start')}, end: {pdt.get('end')}, dur_s: {pdt.get('dur_s'):.3f}, flux: {pdt.get('flux')}")

            st.write("Block detected:", stutter.get("block", False))
            if stutter.get("block_detail"):
                bdt = stutter["block_detail"]
                st.write(f"- details: {bdt}")

            # raw JSON in expander
            st.expander("Show raw result JSON").write(payload)

        status_box.success("Finished")
        # mark running false (in case)
        st.session_state["JOB_STATE"]["running"] = False
    else:
        status_box.write(payload)

# If job is running, force a rerun to keep UI updated (non-blocking because thread does the heavy work).
if st.session_state["JOB_STATE"]["running"]:
    time.sleep(0.2)
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
            st.stop()
    except Exception:
        try:
            st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
        except Exception:
            pass
        st.stop()

# small instructions
st.markdown("""
*Notes*
- An OpenCV window will open for webcam; allow camera access. Press q inside that window to abort early.
- TTS uses system voices (pyttsx3). If you don't have voices for some languages, results may be spoken in default voice.
- Models are loaded from working directory if present. If TensorFlow import or model loading fails, Dysarthria will be skipped.
""")