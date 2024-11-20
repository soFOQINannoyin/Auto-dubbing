import streamlit as st
import whisper  # Correct import for Whisper
import os
from pydub import AudioSegment
import io
from transformers import pipeline

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    # Load whisper model
    model = whisper.load_model("base")
    
    # Save uploaded audio to a temporary file
    audio_path = "uploaded_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    
    # Convert audio to a format Whisper can handle (WAV)
    audio = AudioSegment.from_mp3(audio_path)
    audio.export("converted_audio.wav", format="wav")
    
    # Transcribe the audio
    result = model.transcribe("converted_audio.wav")
    return result['text']

# Function to perform sentiment analysis on the transcribed text
def analyze_sentiment(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(text)
    return sentiment[0]

# Streamlit interface
st.title("Podcast Transcription and Analysis")
st.subheader("Upload your podcast audio file and get the transcription and sentiment analysis!")

# File upload widget
audio_file = st.file_uploader("Upload Podcast Audio", type=["mp3", "wav", "m4a"])

if audio_file:
    # Transcribe audio
    st.write("Processing your podcast...")
    transcription = transcribe_audio(audio_file)
    
    # Display transcription
    st.subheader("Transcription:")
    st.write(transcription)
    
    # Perform Sentiment Analysis
    st.write("Analyzing sentiment of the transcription...")
    sentiment = analyze_sentiment(transcription)
    
    # Display sentiment results
    st.subheader("Sentiment Analysis:")
    st.write(f"Label: {sentiment['label']}")
    st.write(f"Score: {sentiment['score']:.4f}")
    
    # Clean up temporary files
    os.remove("uploaded_audio.mp3")
    os.remove("converted_audio.wav")
