#!/usr/bin/env python3
"""
Video audio extraction utility for MP4, MOV and other video formats.
Extracts audio and converts to WAV format optimized for Whisper.
"""

import sys
import os
import subprocess

def extract_audio_from_video(input_path, output_path):
    """
    Extract audio from video file and convert to WAV format.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path for output WAV file
    """
    try:
        # Use ffmpeg to extract audio with settings optimized for Whisper
        command = [
            "ffmpeg", "-i", input_path,
            "-ac", "1",  # Convert to mono
            "-ar", "16000",  # Set sample rate to 16 kHz
            "-q:a", "0",  # Best quality
            "-map", "a",  # Map audio stream
            output_path, "-y"  # Overwrite output file
        ]
        
        print(f"Extracting audio from {input_path}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            sys.exit(1)
        
        print(f"Successfully extracted audio to {output_path}")
        
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg to process video files.")
        print("Install instructions:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
        sys.exit(1)
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_audio.py <input_video> <output_wav>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    extract_audio_from_video(input_file, output_file)

if __name__ == "__main__":
    main()