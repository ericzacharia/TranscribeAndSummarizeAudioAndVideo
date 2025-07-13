#!/usr/bin/env python3
"""
Enhanced Streamlit web interface for TranscribeAndSummarizeAudioAndVideo.
Supports both audio and video files with multiple transcription and summarization options.
"""

import os
import subprocess
import streamlit as st
import tempfile
import shutil
from pydub import AudioSegment
from openai import OpenAI
from fpdf import FPDF
import markdown2
from io import BytesIO
from datetime import datetime

# Import our utility functions
import sys
sys.path.append(os.path.dirname(__file__))
from summarize_transcript import summarize_general, summarize_meeting

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

# Streamlit page configuration
st.set_page_config(
    page_title="TranscribeAndSummarizeAudioAndVideo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory."""
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir", exist_ok=True)
    
    file_name = uploaded_file.name.replace(" ", "_")
    file_path = os.path.join("tempDir", file_name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def convert_to_wav(input_path, output_path):
    """Convert audio/video file to WAV format."""
    file_ext = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_ext in ['.mp4', '.mov']:
            # Extract audio from video using ffmpeg
            command = [
                "ffmpeg", "-i", input_path,
                "-ac", "1", "-ar", "16000", "-q:a", "0", "-map", "a",
                output_path, "-y"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
        else:
            # Convert audio file using pydub
            if file_ext == ".m4a":
                audio = AudioSegment.from_file(input_path, format="m4a")
            elif file_ext == ".mp3":
                audio = AudioSegment.from_file(input_path, format="mp3")
            elif file_ext == ".wav":
                audio = AudioSegment.from_file(input_path, format="wav")
            else:
                audio = AudioSegment.from_file(input_path)
            
            audio.export(output_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            
    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")

def transcribe_audio(wav_file_path, model_name):
    """Transcribe audio using Whisper.cpp."""
    try:
        projects_path = "/Users/eric/Desktop/2-Career/Projects"
        model_path = f"{projects_path}/models/whisper/ggml-{model_name}.bin"
        
        command = [
            f"{projects_path}/whisper.cpp/main",
            "-m", model_path,
            "-f", wav_file_path,
            "-otxt",
            "-t", "10",      # Use 10 threads for M2 Max
            "--flash-attn",  # Enable flash attention
            "-bs", "1",      # Reduce beam size for speed
            "-bo", "1",      # Reduce best-of candidates to match beam size
            "-mc", "4096",   # Increase max context (plenty of RAM available)
            "-nf"            # No temperature fallback for consistent speed
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Error running whisper.cpp: {result.stderr}")
        
        # Find the transcript file
        transcript_file = wav_file_path.replace(".wav", ".txt")
        fallback_transcript_file = wav_file_path + ".txt"
        
        if os.path.exists(transcript_file):
            with open(transcript_file, "r") as f:
                transcript = f.read()
        elif os.path.exists(fallback_transcript_file):
            with open(fallback_transcript_file, "r") as f:
                transcript = f.read()
        else:
            raise FileNotFoundError("Transcript file not found")
        
        return transcript
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def generate_summary(transcript, style):
    """Generate summary using OpenAI."""
    if not client:
        raise Exception("OpenAI API key not configured")
    
    if style == "meeting":
        return summarize_meeting(transcript)
    else:
        return summarize_general(transcript)

def generate_download_file(content, file_format, filename_base):
    """Generate downloadable file in specified format."""
    if file_format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        
        lines = content.split('\n')
        for line in lines:
            try:
                line = line.replace('•', '-').replace("'", "'").replace('"', '"').replace('"', '"')
                line = line.encode('latin1', 'ignore').decode('latin1')
                pdf.cell(0, 5, txt=line, ln=True)
            except:
                pdf.cell(0, 5, txt="[Line contains unsupported characters]", ln=True)
        
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue(), "application/pdf"
    
    elif file_format == "md":
        return content.encode("utf-8"), "text/markdown"
    
    else:  # txt
        return content.encode("utf-8"), "text/plain"

# Main app interface
st.markdown('<div class="main-header">🎵 TranscribeAndSummarizeAudioAndVideo</div>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
<h3>🚀 Unified Audio & Video Transcription and Summarization</h3>
<p>Upload audio files (M4A, MP3, WAV) or video files (MP4, MOV) to get accurate transcriptions and AI-powered summaries.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_options = {
        "tiny": "Tiny (fastest, least accurate)",
        "base": "Base (balanced)",
        "small": "Small (good accuracy)",
        "medium.en": "Medium English (recommended)",
        "medium": "Medium (multilingual)",
        "large-v3": "Large v3 (best accuracy, slower)"
    }
    
    selected_model = st.selectbox(
        "🤖 Whisper Model",
        options=list(model_options.keys()),
        index=3,  # Default to medium.en
        format_func=lambda x: model_options[x]
    )
    
    # Summary options
    enable_summary = st.checkbox("📋 Enable Summarization", value=True)
    
    if enable_summary:
        summary_style = st.radio(
            "📝 Summary Style",
            options=["general", "meeting"],
            format_func=lambda x: {
                "general": "General (topics, insights, key points)",
                "meeting": "Meeting (action items, JIRA tickets)"
            }[x]
        )
        
        download_format = st.selectbox(
            "💾 Download Format",
            options=["txt", "md", "pdf"],
            format_func=lambda x: {
                "txt": "Text (.txt)",
                "md": "Markdown (.md)", 
                "pdf": "PDF (.pdf)"
            }[x]
        )
    
    # API key status
    if client:
        st.success("✅ OpenAI API configured")
    else:
        st.error("❌ OpenAI API key not found")
        st.info("Set OPENAI_API_KEY in your .env file")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["m4a", "mp3", "wav", "mp4", "mov"],
        help="Supported formats: M4A, MP3, WAV (audio) | MP4, MOV (video)"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / (1024*1024):.1f} MB")
        
        file_type = "🎵 Audio" if uploaded_file.name.split('.')[-1].lower() in ['m4a', 'mp3', 'wav'] else "🎬 Video"
        st.info(f"📋 File type: {file_type}")

with col2:
    st.header("🎯 Processing Status")
    
    if uploaded_file:
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
        
        if process_button:
            with st.spinner("Processing your file..."):
                try:
                    # Save uploaded file
                    input_file_path = save_uploaded_file(uploaded_file)
                    st.success("✅ File saved")
                    
                    # Convert to WAV
                    wav_file_path = input_file_path.rsplit(".", 1)[0] + ".wav"
                    with st.spinner("🔄 Converting to WAV format..."):
                        convert_to_wav(input_file_path, wav_file_path)
                    st.success("✅ Audio conversion completed")
                    
                    # Transcribe
                    with st.spinner(f"🎤 Transcribing with {model_options[selected_model]}..."):
                        transcript = transcribe_audio(wav_file_path, selected_model)
                    st.success("✅ Transcription completed")
                    
                    # Store results in session state
                    st.session_state.transcript = transcript
                    st.session_state.filename = uploaded_file.name.rsplit(".", 1)[0]
                    
                    # Generate summary if enabled
                    if enable_summary and client:
                        with st.spinner(f"🤖 Generating {summary_style} summary..."):
                            summary = generate_summary(transcript, summary_style)
                        st.session_state.summary = summary
                        st.session_state.summary_style = summary_style
                        st.success("✅ Summary generated")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Results section
if 'transcript' in st.session_state:
    st.header("📄 Results")
    
    # Transcript display and download
    with st.expander("📝 Transcript", expanded=True):
        st.text_area(
            "Transcript Content",
            value=st.session_state.transcript,
            height=300,
            key="transcript_display"
        )
        
        # Download transcript
        transcript_data, transcript_mime = generate_download_file(
            st.session_state.transcript, "txt", st.session_state.filename
        )
        st.download_button(
            label="💾 Download Transcript",
            data=transcript_data,
            file_name=f"{st.session_state.filename}_transcript.txt",
            mime=transcript_mime
        )
    
    # Summary display and download
    if 'summary' in st.session_state:
        with st.expander("📋 Summary", expanded=True):
            st.markdown(st.session_state.summary)
            
            # Download summary
            summary_data, summary_mime = generate_download_file(
                st.session_state.summary, download_format, st.session_state.filename
            )
            st.download_button(
                label=f"💾 Download Summary ({download_format.upper()})",
                data=summary_data,
                file_name=f"{st.session_state.filename}_summary_{st.session_state.summary_style}.{download_format}",
                mime=summary_mime
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔧 Powered by Whisper.cpp for transcription and OpenAI for summarization</p>
    <p>💡 Tip: Use 'medium.en' model for best balance of speed and accuracy for English content</p>
</div>
""", unsafe_allow_html=True)