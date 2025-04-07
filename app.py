import streamlit as st
import librosa
import numpy as np
import joblib
import noisereduce as nr

# Load trained model and scaler
model = joblib.load("speech_feedback_model.pkl")
scaler = joblib.load("supervised_scaler.pkl")

def denoise_audio(y, sr):
    return nr.reduce_noise(y=y, sr=sr)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = denoise_audio(y, sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_variance = np.var(mfcc)

        f0, _, _ = librosa.pyin(y, fmin=50, fmax=300)
        f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
        pitch_variation = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0

        rms = librosa.feature.rms(y=y).flatten()
        rms_mean = np.mean(rms)
        energy_stability = np.std(rms)

        pause_threshold = 0.01
        pause_frames = rms <= pause_threshold
        pause_frequency = np.sum(pause_frames)
        pause_duration = np.sum(pause_frames) * (512 / sr)
        speaking_rate = np.sum(rms > pause_threshold)

        articulation_rate = rms_mean

        return [
            f0_mean, pitch_variation, rms_mean, energy_stability,
            pause_frequency, pause_duration, speaking_rate,
            mfcc_variance, articulation_rate
        ]

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def generate_feedback(label):
    responses = {
        "Confident": "✅ Great job! Your speech is fluent and confident.",
        "Moderate": "⚠️ Your speech is moderate. Try improving clarity and energy.",
        "Unclear": "🔁 You sound hesitant. Reduce pauses and increase vocal energy."
    }
    return responses.get(label, "Feedback unavailable.")

# --- Streamlit UI ---
st.title("🗣️ Real-Time Speech Feedback System")
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Analyzing...")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav")

    if features:
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 Predicted Category: **{prediction}**")
        st.info(generate_feedback(prediction))
