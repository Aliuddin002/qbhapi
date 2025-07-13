from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import io
import traceback
import os
import requests
import shutil

# ============ CONSTANTS ============ #
SAMPLE_RATE = 22050
TARGET_SHAPE = (128, 216)  # n_mels × frames (~5 s)
TOP_N_DEFAULT = 5

# ============ MODEL DOWNLOAD SETUP ============ #
MODEL_URL = "https://drive.google.com/uc?export=download&id=1xL8fy4lyvjBARw_EXJHt8h0ChZTqaYQX"
MODEL_PATH = "trained_encoder.h5"

def download_model():
    """Download trained_encoder.h5 if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print("✅ Model already exists.")
        return
    print("⬇️ Downloading trained_encoder.h5 ...")
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH + ".tmp", "wb") as f:
                shutil.copyfileobj(r.raw, f)
        os.rename(MODEL_PATH + ".tmp", MODEL_PATH)
        print("✅ Model downloaded.")
    except Exception as e:
        print("❌ Failed to download model:", e)
        raise

# ============ INITIALISE APP ============ #
app = FastAPI(title="Query‑by‑Humming API",
              description="Returns top matching tracks from FMA-small for a 5-second hum.",
              version="1.0.0")

# ============ LOAD RESOURCES AT START-UP ============ #
try:
    track_df = pd.read_csv("track_df_cleaned.csv")
    features_array = np.load("qbh_features.npy")
    features_index = pd.read_csv("qbh_features_index.csv")

    download_model()
    encoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print("[FATAL] Could not load model or data — check file paths.")
    traceback.print_exc()
    raise e

# ============ HELPER FUNCTIONS ============ #
def audio_to_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=TARGET_SHAPE[0], hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pitches, mags = librosa.piptrack(y=audio, sr=sr, hop_length=512)
    voiced = np.max(mags, axis=0) > np.percentile(mags, 75)
    mel_db[:, voiced] *= 1.5
    if mel_db.shape[1] < TARGET_SHAPE[1]:
        mel_db = np.pad(mel_db, ((0, 0), (0, TARGET_SHAPE[1] - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :TARGET_SHAPE[1]]
    mel_db -= mel_db.min()
    if mel_db.max() > 0:
        mel_db /= mel_db.max()
    return mel_db

def extract_features(mel: np.ndarray) -> np.ndarray:
    mel = np.expand_dims(mel, axis=(0, -1)).astype("float32")
    latent = encoder.predict(mel, verbose=0)[0]
    norm = np.linalg.norm(latent)
    if norm < 1e-6:
        raise ValueError("Degenerate (zero) feature vector")
    return latent / norm

def match_tracks(query_vector, features_array, top_n=5):
    similarities = np.dot(features_array, query_vector)
    similarities /= np.linalg.norm(features_array, axis=1) * np.linalg.norm(query_vector)
    sorted_idx = np.argsort(similarities)[::-1][:top_n]
    results = []
    for rank, i in enumerate(sorted_idx, start=1):
        try:
            track_id = int(features_index.iloc[i]["track_id"])
            row = track_df.loc[track_df["track_id"] == track_id].iloc[0]
            results.append({
                "rank": rank,
                "track_id": track_id,
                "title": str(row["title"]),
                "artist": str(row["artist_name"]),
                "similarity": float(round(similarities[i] * 100, 2))
            })
        except Exception as e:
            print(f"⚠️ Error with index {i}: {e}")
    return results

# ============ ROUTES ============ #
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/qbh")
async def qbh_endpoint(file: UploadFile = File(...), top_n: int = TOP_N_DEFAULT):
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/mpeg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload wav/mp3/flac.")
    try:
        raw = await file.read()
        audio, sr = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True, duration=5.0)
        if audio.size == 0:
            raise ValueError("Empty audio")
        mel = audio_to_mel(audio, sr)
        q_vec = extract_features(mel)
        matches = match_tracks(query_vector=q_vec, features_array=features_array, top_n=top_n)
        return JSONResponse({"matches": matches})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============ MAIN ============ #
if __name__ == "__main__":
    uvicorn.run("qbh_api:app", host="0.0.0.0", port=8000, reload=True)
