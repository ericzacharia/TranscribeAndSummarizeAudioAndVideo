#!/usr/bin/env python3
"""
Audio conversion utility for various audio formats to WAV.
Supports M4A, MP3, and other common audio formats.
"""

import sys
import os
from pydub import AudioSegment

def convert_audio_to_wav(input_path, output_path):
    """
    Convert various audio formats to WAV format optimized for Whisper.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path for output WAV file
    """
    try:
        # Detect input format from file extension
        input_ext = os.path.splitext(input_path)[1].lower()
        
        # Load audio file
        if input_ext == ".m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif input_ext == ".mp3":
            audio = AudioSegment.from_file(input_path, format="mp3")
        elif input_ext == ".wav":
            audio = AudioSegment.from_file(input_path, format="wav")
        else:
            # Try to auto-detect format
            audio = AudioSegment.from_file(input_path)
        
        # Export as WAV with settings optimized for Whisper
        audio.export(
            output_path, 
            format="wav",
            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz sample rate, mono
        )
        
        print(f"Successfully converted {input_path} to {output_path}")
        
    except Exception as e:
        print(f"Error converting audio file: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_audio.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    convert_audio_to_wav(input_file, output_file)

if __name__ == "__main__":
    main()