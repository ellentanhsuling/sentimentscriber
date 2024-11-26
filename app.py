import streamlit as st
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import io
import wave
import numpy as np
from transformers import pipeline
import datetime
import tempfile
import time

# Initialize sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", token=st.secrets["HUGGINGFACE_API_KEY"])

def calculate_risk_level(conversation):
    negative_count = 0
    total_entries = 0
    concerning_phrases = [
        "suicide", "kill", "die", "hurt myself", "end it all", 
        "hopeless", "worthless", "can't go on", "give up"
    ]
    
    risk_indicators = 0
    
    for entry in conversation:
        if entry["speaker"] == "Patient":
            total_entries += 1
            if entry["sentiment"]["label"] == "NEGATIVE":
                negative_count += 1
            
            for phrase in concerning_phrases:
                if phrase in entry["text"].lower():
                    risk_indicators += 1
    
    if total_entries == 0:
        return "Risk Level: Unable to assess (insufficient data)"
    
    negative_ratio = negative_count / total_entries
    
    if negative_ratio > 0.7 or risk_indicators >= 2:
        return "Risk Level: HIGH\nRecommendation: Immediate follow-up recommended"
    elif negative_ratio > 0.4 or risk_indicators == 1:
        return "Risk Level: MODERATE\nRecommendation: Schedule follow-up within 1 week"
    else:
        return "Risk Level: LOW\nRecommendation: Regular scheduled follow-up"

def process_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(44100)
            wave_file.writeframes(audio_bytes)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_file.name) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                return text
            except:
                return None

def detect_speaker(previous_speaker, text):
    counselor_phrases = ["how do you feel", "tell me more", "what do you think", "can you explain"]
    
    if any(phrase in text.lower() for phrase in counselor_phrases) or text.strip().endswith("?"):
        return "Counselor"
    elif previous_speaker == "Counselor":
        return "Patient"
    else:
        return "Counselor"

def main():
    st.title("Counseling Session Transcriber")
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'previous_speaker' not in st.session_state:
        st.session_state.previous_speaker = "Counselor"
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    sentiment_analyzer = load_sentiment_analyzer()
    
    status_placeholder = st.empty()
    audio_level_placeholder = st.empty()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        audio_bytes = audio_recorder(
            pause_threshold=60.0,
            recording_color="#e74c3c",
            neutral_color="#95a5a6",
            icon_name="microphone",
            icon_size="2x"
        )
    
    with col2:
        if audio_bytes:
            st.session_state.recording = True
            status_placeholder.markdown("🔴 **Recording in Progress**")
            
            progress_bar = audio_level_placeholder.progress(0)
            for i in range(10):
                level = np.random.uniform(0.3, 0.9)
                progress_bar.progress(level)
                time.sleep(0.1)
            
            text = process_audio(audio_bytes)
            st.session_state.recording = False
            status_placeholder.markdown("✅ **Recording Complete**")
            audio_level_placeholder.empty()
            
            if text:
                speaker = detect_speaker(st.session_state.previous_speaker, text)
                st.session_state.previous_speaker = speaker
                sentiment = sentiment_analyzer(text)[0]
                st.session_state.conversation.append({
                    "speaker": speaker,
                    "text": text,
                    "sentiment": sentiment
                })
        else:
            if not st.session_state.recording:
                status_placeholder.markdown("⚪ **Ready to Record**")
                audio_level_placeholder.progress(0)
    
    st.markdown("### Conversation Transcript")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['speaker']}**: {entry['text']}")
        st.write(f"Sentiment: {entry['sentiment']['label']} (Score: {entry['sentiment']['score']:.2f})")
        st.write("---")
    
    # Display current risk assessment
    if st.session_state.conversation:
        st.markdown("### Current Risk Assessment")
        st.markdown(calculate_risk_level(st.session_state.conversation))
    
    # Download button with enhanced transcript
    if st.session_state.conversation:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"counseling_session_{timestamp}.txt"
        
        content = "COUNSELING SESSION TRANSCRIPT\n"
        content += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"
        
        for entry in st.session_state.conversation:
            content += f"{entry['speaker']}: {entry['text']}\n"
            content += f"Sentiment: {entry['sentiment']['label']} (Score: {entry['sentiment']['score']:.2f})\n\n"
        
        content += "\nRISK ASSESSMENT\n"
        content += "=" * 50 + "\n"
        content += calculate_risk_level(st.session_state.conversation) + "\n\n"
        
        st.download_button(
            label="Download Transcript",
            data=content,
            file_name=filename,
            mime="text/plain"
        )
    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.previous_speaker = "Counselor"

if __name__ == "__main__":
    main()
