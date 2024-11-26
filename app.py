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

# Add new session state variables
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""

def main():
    st.title("Counseling Session Transcriber")
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'previous_speaker' not in st.session_state:
        st.session_state.previous_speaker = "Counselor"
    
    sentiment_analyzer = load_sentiment_analyzer()
    
    status_placeholder = st.empty()
    audio_level_placeholder = st.empty()
    transcript_placeholder = st.empty()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if not st.session_state.is_recording:
            if st.button("Start Recording"):
                st.session_state.is_recording = True
                st.experimental_rerun()
        else:
            if st.button("Stop Recording"):
                st.session_state.is_recording = False
                st.experimental_rerun()
    
    with col2:
        if st.session_state.is_recording:
            status_placeholder.markdown("🔴 **Recording in Progress**")
            audio_bytes = audio_recorder(
                pause_threshold=2**32,  # Very large number to prevent auto-stop
                recording_color="#e74c3c",
                neutral_color="#95a5a6",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if audio_bytes:
                text = process_audio(audio_bytes)
                if text:
                    speaker = detect_speaker(st.session_state.previous_speaker, text)
                    st.session_state.previous_speaker = speaker
                    sentiment = sentiment_analyzer(text)[0]
                    st.session_state.conversation.append({
                        "speaker": speaker,
                        "text": text,
                        "sentiment": sentiment
                    })
                    # Update live transcript
                    st.session_state.current_transcript = f"**{speaker}**: {text}"
                    
            # Show live transcript
            transcript_placeholder.markdown(st.session_state.current_transcript)
            
            # Show audio level animation
            progress_bar = audio_level_placeholder.progress(0)
            level = np.random.uniform(0.3, 0.9)
            progress_bar.progress(level)
        else:
            status_placeholder.markdown("⚪ **Ready to Record**")
            audio_level_placeholder.progress(0)
            st.session_state.current_transcript = ""
    
    # Rest of your existing code for displaying conversation transcript, 
    # risk assessment, and download button remains the same
    st.markdown("### Conversation Transcript")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['speaker']}**: {entry['text']}")
        st.write(f"Sentiment: {entry['sentiment']['label']} (Score: {entry['sentiment']['score']:.2f})")
        st.write("---")
    
    if st.session_state.conversation:
        st.markdown("### Current Risk Assessment")
        st.markdown(calculate_risk_level(st.session_state.conversation))
        
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
        st.session_state.current_transcript = ""

if __name__ == "__main__":
    main()
