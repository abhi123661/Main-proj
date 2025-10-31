import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------------- Config
CSV_PATH = "parkinsons/parkinsons.csv"   # Your CSV file
MODEL_OUT = "small_model.pkl"
SCALER_OUT = "small_scaler.pkl"
FEATURE_LIST_OUT = "small_features.pkl"

# ------------------------- 1) Load dataset
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print("Loaded dataset with shape:", df.shape)

# ------------------------- 2) Define features to use
features = [
    "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
    "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer",
    "ddaShimmer",
    "meanAutoCorrHarmonicity",  # HNR
    "mean_Log_energy", "std_Log_energy", "mean_delta_log_energy"
]

# MFCC features (mean, std) and delta features
ord_suffix = ["0th", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]

for suf in ord_suffix:
    if f"mean_MFCC_{suf}_coef" in df.columns:
        features.append(f"mean_MFCC_{suf}_coef")
    if f"std_MFCC_{suf}_coef" in df.columns:
        features.append(f"std_MFCC_{suf}_coef")
    if f"mean_{suf}_delta" in df.columns:
        features.append(f"mean_{suf}_delta")

# Only keep features that exist in CSV
features = [c for c in features if c in df.columns]
print(f"Using {len(features)} features: {features}")

if len(features) < 8:
    raise RuntimeError("Too few matching features found in CSV. Check column names.")

# ------------------------- 3) Prepare X, y
X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# Determine target column
if 'class' in df.columns:
    y = df['class'].astype(int)
elif 'status' in df.columns:
    y = df['status'].astype(int)
elif 'target' in df.columns:
    y = df['target'].astype(int)
else:
    raise RuntimeError("No target column found (expected 'class', 'status' or 'target').")

print("Feature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ------------------------- 4) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------- 5) Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------- 6) Train RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_s, y_train)

# ------------------------- 7) Evaluate
y_pred = clf.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=["Healthy","Parkinson's"]))

# ------------------------- 8) Save model, scaler, feature list
joblib.dump(clf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(features, FEATURE_LIST_OUT)

print("Saved model ->", MODEL_OUT)
print("Saved scaler ->", SCALER_OUT)
print("Saved feature list ->", FEATURE_LIST_OUT)
