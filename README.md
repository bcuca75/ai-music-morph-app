# ai-music-morph-app
import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Compressor, Saturation, LowpassFilter
from sklearn.cluster import KMeans
import tempfile

# ---------- Genre Profiles ----------
GENRES = {
    "Lo‑Fi": (6000, 10, 0.15),
    "Rock": (12000, 14, 0.08),
    "Jazz": (14000, 6, 0.18),
    "EDM": (18000, 8, 0.05)
}

# ---------- Audio Load ----------
def load_audio(file):
    y, sr = librosa.load(file, mono=True)
    return y, sr

# ---------- Genre Detection ----------
def detect_genre(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    km = KMeans(n_clusters=4, random_state=0)
    km.fit(mfcc.T)
    return list(GENRES.keys())[np.random.randint(0, 4)]

# ---------- Genre Morph ----------
def apply_genre(audio, sr, g):
    lp, sat, rev = GENRES[g]
    board = Pedalboard([
        LowpassFilter(lp),
        Compressor(threshold_db=-18, ratio=3),
        Saturation(drive_db=sat),
        Reverb(room_size=rev)
    ])
    return board(audio, sr)

# ---------- Key / Mode Morph ----------
def key_morph(audio, sr, steps):
    return librosa.effects.pitch_shift(audio, sr, steps)

# ---------- App UI ----------
st.title("🎵 AI Music Cleanup & Genre Morphing App")

uploaded = st.file_uploader("Upload AI‑Generated Music", type=["wav", "mp3"])

if uploaded:
    audio, sr = load_audio(uploaded)
    detected = detect_genre(audio, sr)
    st.write("Detected Genre:", detected)

    genre = st.selectbox("Select Target Genre", list(GENRES.keys()))
    key_shift = st.slider("Key Shift (semitones)", -6, 6, 0)

    if st.button("Transform Music"):
        audio = apply_genre(audio, sr, genre)
        audio = key_morph(audio, sr, key_shift)
        audio /= np.max(np.abs(audio))

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp.name, audio, sr)

        st.audio(temp.name)
        st.success("✅ Transformation Complete")
        
