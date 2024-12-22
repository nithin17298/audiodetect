import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from transformers import pipeline
import pyaudio
import wave
import os

st.title("Real-Time Audio Detection with Emotion Classification")
st.write("Record your voice and analyze emotions as Positive or Negative.")

# Directory to save recorded audio
record_dir = "recordings"
os.makedirs(record_dir, exist_ok=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
audio_filename = os.path.join(record_dir, "realtime_audio.wav")

# Real-time recording function
def record_audio():
    st.write("Recording...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Save the recorded audio
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    st.write("Recording complete!")

# Emotion recognition model
emotion_recognition = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Mapping emotions to Positive/Negative
positive_emotions = {"joy", "love"}
negative_emotions = {"anger", "sadness", "fear", "disgust", "surprise"}

# Record and analyze audio
if st.button("Record Audio"):
    record_audio()
    st.audio(audio_filename, format='audio/wav')

    # Load the audio for analysis
    y, sr = librosa.load(audio_filename, sr=None)
    st.write(f"Sample Rate: {sr}")
    st.write(f"Audio Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")

    # Display waveform
    st.write("Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    plt.title("Waveform")
    st.pyplot(fig)

    # Spectrogram
    st.write("Spectrogram:")
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    st.pyplot(fig)

    # Emotion analysis
    st.write("Analyzing emotions...")
    # Dummy text input for demonstration
    transcript = "This is a placeholder for speech-to-text transcription."
    emotion_results = emotion_recognition(transcript)

    # Display results as Positive/Negative
    emotion_labels = [emotion["label"] for emotion in emotion_results]
    detected_positive = [e for e in emotion_labels if e in positive_emotions]
    detected_negative = [e for e in emotion_labels if e in negative_emotions]

    if detected_positive:
        st.write("Overall Sentiment: **Positive**")
        st.write(f"Detected Positive Emotions: {', '.join(detected_positive)}")
    elif detected_negative:
        st.write("Overall Sentiment: **Negative**")
        st.write(f"Detected Negative Emotions: {', '.join(detected_negative)}")
    else:
        st.write("Overall Sentiment: **Neutral**")
