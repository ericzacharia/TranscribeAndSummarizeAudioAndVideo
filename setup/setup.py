#!/usr/bin/env python3

"""
TranscribeAndSummarizeAudioAndVideo Setup Script
Cross-platform setup for the transcription and summarization tool
"""

import os
import sys
import stat
import shutil
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🚀 Setting up TranscribeAndSummarizeAudioAndVideo...")
    print()

def make_scripts_executable():
    """Make Python scripts executable on Unix-like systems"""
    scripts = [
        "run.py",
        "src/summarize_transcript.py",
        "src/convert_audio.py", 
        "src/extract_audio.py"
    ]
    
    if platform.system() in ['Darwin', 'Linux']:  # macOS and Linux
        for script in scripts:
            script_path = Path(script)
            if script_path.exists():
                # Add execute permission
                current_mode = script_path.stat().st_mode
                script_path.chmod(current_mode | stat.S_IEXEC)
        print("✅ Made scripts executable")
    else:
        print("✅ Scripts ready (Windows - no chmod needed)")

def create_directories():
    """Create necessary output directories"""
    directories = [
        "inputs",
        "outputs",
        "docs/generated_reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created output directories")

def setup_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists():
        if env_template.exists():
            shutil.copy2(env_template, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and add your OpenAI API key")
        else:
            print("❌ .env.template not found")
    else:
        print("✅ .env file already exists")

def check_python():
    """Check Python installation"""
    try:
        version = sys.version.split()[0]
        print(f"✅ Python found: {version}")
        
        # Check if Python version is 3.8+
        major, minor = map(int, version.split('.')[:2])
        if major < 3 or (major == 3 and minor < 8):
            print("⚠️  Python 3.8+ recommended for best compatibility")
        
        return True
    except Exception:
        print("❌ Python not found or version check failed")
        return False

def check_ffmpeg():
    """Check FFmpeg installation"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Get first line which contains version info
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg found: {version_line}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("❌ FFmpeg not found. Install with:")
    system = platform.system()
    if system == "Darwin":  # macOS
        print("   macOS: brew install ffmpeg")
    elif system == "Linux":
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   CentOS/RHEL: sudo yum install ffmpeg")
    elif system == "Windows":
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   Or use chocolatey: choco install ffmpeg")
    
    return False

def check_whisper_cpp():
    """Check for Whisper.cpp installation"""
    local_whisper = Path("./whisper.cpp/main")
    external_whisper = Path("/Users/eric/Desktop/2-Career/Projects/whisper.cpp/main")
    
    # Check for Windows executable
    if platform.system() == "Windows":
        local_whisper = Path("./whisper.cpp/main.exe")
        external_whisper = None  # Skip external path on Windows
    
    if local_whisper.exists():
        print("✅ Whisper.cpp found locally at ./whisper.cpp/")
        return True
    elif external_whisper and external_whisper.exists():
        print(f"⚠️  Whisper.cpp found at external location: {external_whisper}")
        print("   Consider copying to local ./whisper.cpp/ directory")
        print(f"   Run: cp -r {external_whisper.parent} ./whisper.cpp")
        return True
    else:
        print("❌ Whisper.cpp not found")
        print("   Option 1: Download prebuilt binary from https://github.com/ggerganov/whisper.cpp")
        print("   Option 2: Build from source (see whisper.cpp documentation)")
        if external_whisper:
            print(f"   Option 3: Copy existing installation:")
            print(f"     cp -r {external_whisper.parent} ./whisper.cpp")
        return False

def check_ollama():
    """Check for Ollama installation (optional)"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama found: {version}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("⚠️  Ollama not found (optional for local AI)")
    print("   Install from: https://ollama.ai")
    print("   Then run: ollama pull llama3.1")
    return False

def print_next_steps():
    """Print next steps for the user"""
    print()
    print("📦 Install Python dependencies:")
    
    # Check if we're in a conda environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print("   Dependencies should already be installed via conda")
    else:
        print("   pip install -r requirements.txt")
        print("   OR use conda: conda env create -f environment.yml")
    
    print()
    print("🔑 Configure API keys:")
    print("   1. Edit .env file")
    print("   2. Add your OpenAI API key (for default AI provider)")
    print("   3. Or install Ollama for local AI models")
    
    print()
    print("🎯 Usage examples:")
    print("   python run.py audio.m4a")
    print("   python run.py video.mp4 --style meeting --format md")
    print("   streamlit run src/app.py")
    
    print()
    print("✨ Setup complete!")

def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Make scripts executable
    make_scripts_executable()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Setup .env file
    setup_env_file()
    
    # Step 4: Check dependencies
    print()
    print("🔍 Checking dependencies...")
    
    python_ok = check_python()
    ffmpeg_ok = check_ffmpeg()
    whisper_ok = check_whisper_cpp()
    ollama_ok = check_ollama()
    
    # Summary
    print()
    print("📋 Dependency Summary:")
    print(f"   Python: {'✅' if python_ok else '❌'}")
    print(f"   FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
    print(f"   Whisper.cpp: {'✅' if whisper_ok else '❌'}")
    print(f"   Ollama: {'✅' if ollama_ok else '⚠️ '} (optional)")
    
    if not ffmpeg_ok or not whisper_ok:
        print()
        print("⚠️  Some required dependencies are missing.")
        print("   Please install them before using the tool.")
    
    # Next steps
    print_next_steps()

if __name__ == "__main__":
    main()