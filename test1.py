"""
audio.py
Unified Speech Health Detector - Multi-Sentence & Real-Time
Multilingual Version: English, Hindi, Kannada, Telugu, Spanish
Detects Parkinson's, Dysarthria, and Stuttering
"""
import os
import io
import numpy as np
import streamlit as st
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
from tensorflow.keras.models import load_model
import python_speech_features as psf
import random

# ---------------- CONFIG ----------------
SR = 16000
DEFAULT_RECORD_SECONDS = 12
DYS_N_MFCC = 16

PARK_MODEL_FILE = "small_model.pkl"
PARK_SCALER_FILE = "small_scaler.pkl"
PARK_FEATURES_FILE = "small_features.pkl"
DYS_MODEL_FILE = "best_model_mfcc_delta.keras"
DYS_SCALER_FILE = "scaler_mfcc_delta.save"

# ---------------- MULTILINGUAL ----------------
LANGUAGES = ["en", "hi", "kn", "te", "es"]
CURRENT_LANG = "en"

TRANSLATIONS = {
    "title": {
        "en": "Unified Speech Health Detector - Multi-Sentence Reading",
        "hi": "एकीकृत भाषण स्वास्थ्य डिटेक्टर - बहु-वाक्य पठन",
        "kn": "ಸಂಯುಕ್ತ ಭಾಷಣ ಆರೋಗ್ಯ ಪತ್ತೆಗಾರ - ಬಹು-ವಾಕ್ಯ ಓದು",
        "te": "సంకలిత వాయిస్ ఆరోగ్య గుర్తింపు - బహు వాక్య పఠనం",
        "es": "Detector de Salud del Habla Unificado - Lectura de Múltiples Oraciones"
    },
    "record_button": {"en":"Run Detection","hi":"डिटेक्शन चलाएँ","kn":"ಪರೀಕ್ಷೆ ನಡೆಸಿ","te":"డిటెక్షన్ నిర్వహించండి","es":"Ejecutar Detección"},
    "audio_source": {"en":"Audio source","hi":"ऑडियो स्रोत","kn":"ಧ್ವನಿ ಮೂಲ","te":"ఆడియో మూలం","es":"Fuente de audio"},
    "upload_warning": {"en":"Please upload WAV file","hi":"कृपया WAV फ़ाइल अपलोड करें","kn":"ದಯವಿಟ್ಟು WAV ಫೈಲ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ","te":"దయచేసి WAV ఫైల్ అప్‌లోడ్ చేయండి","es":"Por favor suba un archivo WAV"},
    "read_sentence": {"en":"Please read aloud the following sentence","hi":"कृपया निम्न वाक्य जोर से पढ़ें","kn":"ದಯವಿಟ್ಟು ಕೆಳಗಿನ ವಾಕ್ಯವನ್ನು ಉಚ್ಚಾರಿಸಿ ಓದಿರಿ","te":"దయచేసి క్రింది వాక్యాన్ని స్వరంగా చదవండి","es":"Por favor lea en voz alta la siguiente oración"},
    "overall_parkinson": {"en":"Overall Parkinson's →","hi":"कुल पार्किंसंस →","kn":"ಒಟ್ಟು ಪಾರ್ಕಿನ್ಸನ್ →","te":"మొత్తం పార్కిన్సన్స్ →","es":"Parkinson general →"},
    "overall_dysarthria": {"en":"Overall Dysarthria →","hi":"कुल डिसार्थ्रिया →","kn":"ಒಟ್ಟು ಡಿಸಾರ್ಥ್ರಿಯಾ →","te":"మొత్తం డిసార్త్రియా →","es":"Disartria general →"},
    "sentence_stutter": {"en":"Sentence {i} Stutter:","hi":"वाक्य {i} हकलाना:","kn":"ವಾಕ್ಯ {i} ಹತ್ತಿರದ ಮಾತು:","te":"వాక్యం {i} మాటల గందరగోళం:","es":"Oración {i} Tartamudez:"},
    "healthy": {"en":"Healthy","hi":"स्वस्थ","kn":"ಆರೋಗ್ಯವಂತ","te":"ఆరోగ్యవంతుడు","es":"Saludable"},
    "parkinsons": {"en":"Parkinson's","hi":"पार्किंसंस","kn":"ಪಾರ್ಕಿನ್ಸನ್","te":"పార్కిన్సన్స్","es":"Parkinson"},
    "dysarthria": {"en":"Dysarthria","hi":"डिसार्थ्रिया","kn":"ಡಿಸಾರ್ಥ್ರಿಯಾ","te":"డిసార్త్రియా","es":"Disartria"},
    "non_dysarthria": {"en":"Non-Dysarthria","hi":"गैर-डिसार्थ्रिया","kn":"ನಾನ್-ಡಿಸಾರ್ಥ್ರಿಯಾ","te":"నాన్-డిసార్త్రియా","es":"No Disartria"}
}

def t(key: str) -> str:
    return TRANSLATIONS.get(key, {}).get(CURRENT_LANG, TRANSLATIONS.get(key, {}).get("en", key))

# ---------------- UTILS ----------------

import vosk
import json

# Load Vosk model (download one first, e.g., small-en-us)
vosk_model = vosk.Model("vosk-model-small-en-us-0.15")


def speech_to_text(y, sr=16000):
    """
    Convert audio array y to text using Vosk
    """
    import wave
    import tempfile

    # Save temporarily to WAV
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmpfile.name, y, sr)

    wf = wave.open(tmpfile.name, "rb")
    rec = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
    text_result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text_result += " " + res.get("text", "")
    # Final partial result
    res = json.loads(rec.FinalResult())
    text_result += " " + res.get("text", "")

    return text_result.strip()


def save_wav_array(path, y, sr=SR):
    y = y.astype(np.float32)
    max_abs = np.max(np.abs(y))
    if max_abs>1: y/=max_abs
    sf.write(path, y, sr, subtype='PCM_16')

def record_audio(duration=DEFAULT_RECORD_SECONDS, sr=SR, channels=1):
    st.info(f"Recording {duration} s ...")
    recording = sd.rec(int(duration*sr), samplerate=sr, channels=channels, dtype='float32')
    sd.wait()
    if channels>1: recording=np.mean(recording,axis=1)
    max_abs=np.max(np.abs(recording))
    if max_abs>0: recording/=max_abs
    return recording

def plot_spectrogram_from_array(y,sr=SR):
    D=librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max+1e-9)
    plt.figure(figsize=(8,3))
    librosa.display.specshow(D,sr=sr,x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    return plt

# ---------------- LOAD MODELS ----------------
park_model = park_scaler = park_feature_list = None
dys_model = dys_scaler = dys_expected_features = None

def safe_load_models():
    global park_model, park_scaler, park_feature_list
    global dys_model, dys_scaler, dys_expected_features
    if all(os.path.exists(f) for f in [PARK_MODEL_FILE,PARK_SCALER_FILE]):
        park_model=joblib.load(PARK_MODEL_FILE)
        park_scaler=joblib.load(PARK_SCALER_FILE)
        if os.path.exists(PARK_FEATURES_FILE): park_feature_list=list(joblib.load(PARK_FEATURES_FILE))
    if all(os.path.exists(f) for f in [DYS_MODEL_FILE,DYS_SCALER_FILE]):
        try:
            dys_model=load_model(DYS_MODEL_FILE, compile=False)  # important fix
        except:
            dys_model=None
        dys_scaler=joblib.load(DYS_SCALER_FILE)
        dys_expected_features=getattr(dys_scaler,"n_features_in_",None)

safe_load_models()

# ---------------- PARKINSON'S ----------------
def compute_parkinson_features(y,sr=SR):
    y=np.asarray(y,dtype=np.float32).flatten()
    if np.max(np.abs(y))>0: y/=np.max(np.abs(y))
    try: y,_=librosa.effects.trim(y,top_db=30)
    except: pass
    if y.size<200: y=np.pad(y,(0,max(0,200-y.size)))
    try: mfcc=psf.mfcc(y,samplerate=sr,numcep=13); mfcc_mean=np.mean(mfcc,axis=0); mfcc_std=np.std(mfcc,axis=0)
    except: mfcc_mean=np.zeros(13); mfcc_std=np.zeros(13)
    try: zcr=float(np.mean(librosa.feature.zero_crossing_rate(y)))
    except: zcr=0
    try: spec_cent=float(np.mean(librosa.feature.spectral_centroid(y=y,sr=sr)))
    except: spec_cent=0
    try: spec_bw=float(np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr)))
    except: spec_bw=0
    try: rms=float(np.mean(librosa.feature.rms(y=y)))
    except: rms=0
    try: pitches,mags=librosa.piptrack(y=y,sr=sr); pitches=pitches[mags>np.median(mags)]; pitch_mean=float(np.mean(pitches)) if pitches.size>0 else 0; pitch_std=float(np.std(pitches)) if pitches.size>0 else 0
    except: pitch_mean,pitch_std=0,0
    features=np.concatenate([mfcc_mean,mfcc_std,[zcr,spec_cent,spec_bw,rms,pitch_mean,pitch_std]])
    features=np.nan_to_num(features,nan=0.0,posinf=0.0,neginf=0.0)
    return features.astype(np.float32)

def adapt_features_to_scaler_and_model(features):
    global park_scaler
    features = np.nan_to_num(features)
    if park_scaler is not None:
        n_expected = park_scaler.n_features_in_
        if features.size < n_expected:
            features = np.concatenate([features, np.zeros(n_expected - features.size)])
        elif features.size > n_expected:
            features = features[:n_expected]
        features = park_scaler.transform(features.reshape(1, -1))
    return features

def predict_parkinson_from_array(y):
    features=compute_parkinson_features(y)
    features=adapt_features_to_scaler_and_model(features)
    if park_model is None: return {"label":None,"prob":None}
    prob=park_model.predict_proba(features)[0][1]
    label=int(prob>0.4)
    return {"label":label,"prob":prob}

# ---------------- DYSARTHRIA ----------------
def preprocess_audio_for_dys(y, sr=SR):
    y = np.asarray(y, dtype=np.float32).flatten()
    if np.max(np.abs(y)) > 0: y /= np.max(np.abs(y))
    try:
        mfcc = psf.mfcc(y, samplerate=sr, numcep=DYS_N_MFCC)
        delta = psf.delta(mfcc, 1)
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(delta, axis=1)
        features = np.concatenate([mfcc_mean, delta_mean])
    except:
        features = np.zeros(DYS_N_MFCC * 2)
    return features.astype(np.float32)

def predict_dys_from_array(y):
    features = preprocess_audio_for_dys(y)
    if dys_scaler is not None:
        n_expected = dys_scaler.n_features_in_
        if features.size < n_expected:
            features = np.concatenate([features, np.zeros(n_expected - features.size)])
        elif features.size > n_expected:
            features = features[:n_expected]
        features = dys_scaler.transform(features.reshape(1, -1))
    features_cnn = features.reshape(1, features.shape[1], 1, 1).astype(np.float32)
    if dys_model is None:
        return 0.0
    prob = float(dys_model.predict(features_cnn))
    return prob

# ---------------- REAL STUTTER ----------------
def extract_stutter_features(y, sr=SR):
    y = np.asarray(y, dtype=np.float32).flatten()
    if np.max(np.abs(y)) > 0: y /= np.max(np.abs(y))

    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)  # 10ms

    # Energy per frame
    energy = np.array([np.sum(np.abs(y[i:i + frame_length] ** 2))
                       for i in range(0, len(y) - frame_length, hop_length)])

    # ZCR per frame
    zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y[i:i + frame_length].reshape(1, -1)))
                    for i in range(0, len(y) - frame_length, hop_length)])

    # Silence threshold: frames with energy < 10% max
    silence_thresh = 0.1 * np.max(energy)
    silent_frames = np.sum(energy < silence_thresh)
    total_frames = len(energy)
    silence_ratio = silent_frames / (total_frames + 1e-9)

    # High ZCR bursts (possible repetitions)
    high_zcr_frames = np.sum(zcr > 0.1)
    repetition_ratio = high_zcr_frames / (total_frames + 1e-9)

    # Combine into stutter probability (weighted sum)
    stutter_prob = np.clip(0.5 * repetition_ratio + 0.5 * silence_ratio, 0, 1)
    return stutter_prob


def run_stutter_detection_on_array(y, sr=SR, ref_text=""):
    spoken_text = speech_to_text(y, sr)

    # Compute stutter probability based on audio
    audio_prob = extract_stutter_features(y, sr)

    # Compute text-based disfluency (simple example: repeated words)
    words_spoken = spoken_text.split()
    words_ref = ref_text.split()
    repeats = sum(1 for i in range(1, len(words_spoken)) if words_spoken[i] == words_spoken[i - 1])
    repeat_ratio = repeats / (len(words_spoken) + 1e-9)

    # Combine audio + text features
    stutter_prob = np.clip(0.6 * audio_prob + 0.4 * repeat_ratio, 0, 1)
    label = "Stutter" if stutter_prob >= 0.5 else "No stutter"
    return f"{label} (prob: {stutter_prob:.3f})", spoken_text


# ---------------- MULTI-SENTENCE ----------------
def run_multi_sentence(y, ref_text=""):
    segments=[y]
    parkinson_probs=[predict_parkinson_from_array(seg) for seg in segments]
    dys_probs=[predict_dys_from_array(seg) for seg in segments]
    stutter_results=[run_stutter_detection_on_array(seg, ref_text=ref_text) for seg in segments]

    agg={"parkinson_label":int(np.mean([p["label"] for p in parkinson_probs if p["label"] is not None])>0.5) if parkinson_probs else None,
         "parkinson_prob":float(np.mean([p["prob"] for p in parkinson_probs if p["prob"] is not None])) if parkinson_probs else None,
         "dysarthria_prob":float(np.mean(dys_probs)) if dys_probs else None,
         "stutter_result":stutter_results}
    return agg, segments

# ---------------- STREAMLIT UI ----------------
CURRENT_LANG=st.selectbox("Select language", LANGUAGES, format_func=lambda x:x.upper())
st.title(t("title"))

SENTENCES_DICT={
    "en":["The quick brown fox jumps over the lazy dog.","She sells seashells by the seashore.","Reading aloud helps detect speech issues.","Consistency is key for accurate detection."],
    "hi":["तेज़ भूरी लोमड़ी आलसी कुत्ते पर कूदती है।","वह समुद्र तट पर शंख बेचती है।","जोर से पढ़ने से भाषण समस्याओं का पता लगता है।","सटीक पहचान के लिए निरंतरता महत्वपूर्ण है।"],
    "kn":["ತ್ವರಿತ ಬೂದು ನಾಯಿ ಮೇಲೆ ಜಿಗಿತಕ್ಕೆ ಹಾರುತ್ತದೆ.","ಅವಳು ಕಡಲತೀರದಲ್ಲಿ ಶಂಕು ಮಾರುತ್ತಾಳೆ.","ಎಚ್ಚರಿಕೆಯಿಂದ ಓದುವುದರಿಂದ ಭಾಷಣ ಸಮಸ್ಯೆಗಳನ್ನು ಪತ್ತೆ ಮಾಡಬಹುದು.","ಖಚಿತ ಪತ್ತೆಗಾಗಿ ಸ್ಥಿರತೆ ಮುಖ್ಯ."],
    "te":["వేగంగా గోధుమ నక్క ఆలస్య కుక్క పై కదులుతుంది.","ఆమె సముద్రతీరంలో శంఖాలు అమ్ముతుంది.","ఎచ్చరికగా చదవడం మాట సమస్యలను గుర్తించడంలో సహాయపడుతుంది.","ఖచ్చిత గుర్తింపుకు స్థిరత్వం కీలకం."],
    "es":["El rápido zorro marrón salta sobre el perro perezoso.","Ella vende conchas en la orilla del mar.","Leer en voz alta ayuda a detectar problemas del habla.","La consistencia es clave para una detección precisa."]
}

sentence_to_read=random.choice(SENTENCES_DICT[CURRENT_LANG])
st.markdown(f"{t('read_sentence')}:\n\n> {sentence_to_read}")

option=st.selectbox(t("audio_source"),["Record (microphone)","Upload WAV"])
duration=st.number_input("Record duration (s)", min_value=5, max_value=60, value=12)
ref_text=st.text_area("Reference text for stutter detection", value=sentence_to_read)
show_spect=st.checkbox("Show spectrogram", value=True)
uploaded_file=None
if option=="Upload WAV": uploaded_file=st.file_uploader("Upload WAV file", type=["wav"])
record_button=st.button(t("record_button"))
result_placeholder=st.empty()
spectro_placeholder=st.empty()

def process_and_display(y_array, ref_text_local=""):
    if show_spect:
        try:
            plt_fig=plot_spectrogram_from_array(y_array)
            spectro_placeholder.pyplot(plt_fig)
            plt.close("all")
        except Exception: pass
    agg,res_per_sent=run_multi_sentence(y_array, ref_text_local)
    out=[]
    if agg.get("parkinson_label") is not None:
        lab=t("parkinsons") if agg['parkinson_label']==1 else t("healthy")
        if agg.get("parkinson_prob") is not None:
            out.append(f"{t('overall_parkinson')} {lab} (confidence: {agg['parkinson_prob']:.3f})")
        else:
            out.append(f"{t('overall_parkinson')} {lab} (confidence: unknown)")
    else: out.append(f"{t('overall_parkinson')} (no result)")
    if agg.get("dysarthria_prob") is not None:
        lab=t("dysarthria") if agg['dysarthria_prob']>=0.5 else t("non_dysarthria")
        out.append(f"{t('overall_dysarthria')} {lab} (prob: {agg['dysarthria_prob']:.3f})")
    else: out.append(f"{t('overall_dysarthria')} (no result)")
    if agg.get("stutter_result") is not None:
        for i,sr in enumerate(agg["stutter_result"]):
            out.append(t("sentence_stutter").format(i=i+1)+f" {sr}")
    result_placeholder.markdown("\n\n".join(out))

if record_button:
    try:
        if option=="Record (microphone)":
            y_array=record_audio(duration)
        elif uploaded_file is not None:
            y_array,sr=librosa.load(io.BytesIO(uploaded_file.read()), sr=SR)
        else:
            st.warning(t("upload_warning"))
            y_array=None
        if y_array is not None:
            process_and_display(y_array, ref_text)
    except Exception as e:
        st.error(f"Error: {str(e)}")