"""
FastAPI backend for Query‑by‑Humming (QbH)
=========================================
• Accepts an uploaded 5‑second hum (wav or mp3)
• Returns the top‑N matching FMA‑small tracks
• Loads encoder and pre‑computed embeddings once at start‑up

How to run locally (for testing):
    uvicorn qbh_api:app --host 0.0.0.0 --port 8000 --reload

Deploy to Google Cloud Run:
1. Put this file plus the model/data files in the same folder
2. Add the Dockerfile shown in the README below
3. gcloud builds submit --tag gcr.io/<PROJECT_ID>/qbh-api
4. gcloud run deploy qbh-api --image gcr.io/<PROJECT_ID>/qbh-api --platform=managed \
       --allow-unauthenticated --memory=1Gi --cpu=1

Then call it from Firebase Studio frontend with fetch()/axios.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import io
import traceback

# ============ CONSTANTS ============ #
SAMPLE_RATE = 22050
TARGET_SHAPE = (128, 216)  # n_mels × frames (~5 s)
TOP_N_DEFAULT = 5

# ============ INITIALISE APP ============ #
app = FastAPI(title="Query‑by‑Humming API",
              description="Returns top matching tracks from FMA‑small for a 5‑second hum.",
              version="1.0.0")

# ============ LOAD RESOURCES AT START‑UP ============ #
# These load once when the container starts, so each request is fast.

try:
    track_df = pd.read_csv("track_df_cleaned.csv")
    features_array = np.load("qbh_features.npy")  # shape (N, latent_dim)
    features_index = pd.read_csv("qbh_features_index.csv")  # columns: track_id
    encoder = tf.keras.models.load_model("trained_encoder.h5", compile=False)
except Exception as e:
    print("[FATAL] Could not load model or data — check file paths.")
    traceback.print_exc()
    raise e

# ============ HELPER FUNCTIONS ============ #

def audio_to_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert mono audio to normalised mel log‑power spectrogram of fixed size."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=TARGET_SHAPE[0], hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Voice emphasise (optional)
    pitches, mags = librosa.piptrack(y=audio, sr=sr, hop_length=512)
    voiced = np.max(mags, axis=0) > np.percentile(mags, 75)
    mel_db[:, voiced] *= 1.5

    # Pad or crop to TARGET_SHAPE[1]
    if mel_db.shape[1] < TARGET_SHAPE[1]:
        mel_db = np.pad(mel_db, ((0, 0), (0, TARGET_SHAPE[1] - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :TARGET_SHAPE[1]]

    # Min‑max normalise 0‑1
    mel_db -= mel_db.min()
    if mel_db.max() > 0:
        mel_db /= mel_db.max()
    return mel_db


def extract_features(mel: np.ndarray) -> np.ndarray:
    """Pass mel through encoder and L2‑normalise."""
    mel = np.expand_dims(mel, axis=(0, -1)).astype("float32")
    latent = encoder.predict(mel, verbose=0)[0]
    norm = np.linalg.norm(latent)
    if norm < 1e-6:
        raise ValueError("Degenerate (zero) feature vector")
    return latent / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_tracks(query_vector, features_array, top_n=5):
    similarities = np.dot(features_array, query_vector)  # Vectorized cosine similarity
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
    """Receive an audio file, return top‑N matches."""
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/mpeg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload wav/mp3/flac.")

    try:
        raw = await file.read()
        audio, sr = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True, duration=5.0)
        if audio.size == 0:
            raise ValueError("Empty audio")
        mel = audio_to_mel(audio, sr)
        q_vec = extract_features(mel)
        matches = match_tracks(query_vector=q_vec, features_array=features_array, top_n=5)
        return JSONResponse({"matches": matches})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============ MAIN (for local run) ============ #
if __name__ == "__main__":
    uvicorn.run("qbh_api:app", host="0.0.0.0", port=8000, reload=True)
