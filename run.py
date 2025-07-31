#!/usr/bin/env python3

"""
TranscribeAndSummarizeAudioAndVideo - Unified CLI Script
Supports M4A, MP4, MOV, WAV files with transcription and summarization
"""

import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime
import re
import urllib.parse

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Import merge functionality
from src.merge_media import find_all_audio_video_files, merge_files, organize_segments, extract_base_name

def show_usage():
    """Show detailed usage information"""
    print("TranscribeAndSummarizeAudioAndVideo - Unified transcription and summarization tool")
    print("")
    print("Usage: python run.py <input_file_or_directory_or_youtube_url> [OPTIONS]")
    print("")
    print("Supported inputs:")
    print("  Audio files: .m4a, .wav, .mp3")
    print("  Video files: .mp4, .mov")
    print("  YouTube URLs: https://youtube.com/watch?v=... or https://youtu.be/...")
    print("  Directories: Process all audio/video files in a directory")
    print("")
    print("OPTIONS:")
    print("  --transcription-model MODEL    Available: tiny, base, small, medium, medium.en (default), large-v3, large-v3-q5_0")
    print("  --summarization-model MODEL    Available: gpt-4o (default), gpt-4o-mini, llama3.2:1b, llama3.2:3b, llama3.1:8b, llama3.1:70b")
    print("  --style STYLE                  Summary style: smart (intelligent multi-category, default), general, meeting")
    print("  --format FORMAT                Output format: md (default), txt, pdf, json")
    print("  --confidence-threshold FLOAT   Minimum confidence for category inclusion (default: 0.5)")
    print("  --show-classification          Display content classification results")
    print("  --interactive                  Ask for user input when classification is uncertain")
    print("  --context-hints HINTS          Provide context hints for ambiguous terms (format: term1=meaning1,term2=meaning2)")
    print("  --help, -h                     Show this help message")
    print("")
    print("Examples:")
    print("  # Single file processing")
    print("  python run.py inputs/voice_memo.m4a                              # Default: medium.en + gpt-4o")
    print("  python run.py inputs/lecture.mp4 --style general --format md     # Custom style and format")
    print("  python run.py inputs/meeting.mp4 --summarization-model llama3.2:3b  # Use local Ollama model")
    print("  python run.py inputs/interview.m4a --transcription-model large-v3    # High quality transcription
  python run.py inputs/meeting.m4a --interactive                       # Interactive smart analysis
  python run.py inputs/technical.m4a --context-hints "entities=data_objects,opsec=security"  # Context hints")
    print("")
    print("  # YouTube URL processing")
    print("  python run.py https://www.youtube.com/watch?v=QT6T6AC02-Q       # Transcribe YouTube video")
    print("  python run.py https://youtu.be/QT6T6AC02-Q --style meeting       # YouTube with meeting format")
    print("")
    print("  # Directory batch processing")
    print("  python run.py inputs/2025-07-15/                                 # Process all files in directory")
    print("  python run.py inputs/ --style meeting --format pdf               # Batch process with custom options")
    print("")
    print("Output locations:")
    print("  Audio files: ./inputs/")
    print("  Session outputs: ./outputs/YYYY-MM-DD/[AM|PM]HH.MM.SS/")
    print("  Models: ./models/ (auto-downloaded)")
    print("")

def validate_file_extension(filename):
    """Validate that the file has a supported extension"""
    extension = Path(filename).suffix.lower().lstrip('.')
    supported_extensions = ['m4a', 'wav', 'mp3', 'mp4', 'mov']
    
    if extension not in supported_extensions:
        print(f"Error: Unsupported file type: .{extension}")
        print(f"Supported types: {', '.join(supported_extensions)}")
        sys.exit(1)
    
    return extension

def validate_summary_style(style):
    """Validate summary style"""
    valid_styles = ['general', 'meeting', 'smart']
    if style not in valid_styles:
        print(f"Error: Invalid summary style: {style}")
        print(f"Supported styles: {', '.join(valid_styles)}")
        sys.exit(1)

def validate_output_format(format_type):
    """Validate output format"""
    valid_formats = ['txt', 'md', 'pdf', 'json']
    if format_type not in valid_formats:
        print(f"Error: Invalid output format: {format_type}")
        print(f"Supported formats: {', '.join(valid_formats)}")
        sys.exit(1)

def find_whisper_path():
    """Find whisper.cpp installation path"""
    # Check local installation
    local_path = SCRIPT_DIR / "whisper.cpp"
    
    if local_path.exists() and (local_path / "build" / "bin" / "whisper-cli").exists():
        return local_path
    else:
        print("Error: whisper.cpp not found. Please run the setup script:")
        print("  python setup/setup.py")
        sys.exit(1)

def create_output_directory():
    """Create timestamped output directory for this processing session"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    
    # Format time as AM00.00.00 - AM11.59.59 or PM00.00.00 - PM11.59.59
    hour = now.hour
    minute = now.minute
    second = now.second
    
    if hour == 0:
        time_str = f"AM12.{minute:02d}.{second:02d}"
    elif hour < 12:
        time_str = f"AM{hour:02d}.{minute:02d}.{second:02d}"
    elif hour == 12:
        time_str = f"PM12.{minute:02d}.{second:02d}"
    else:
        time_str = f"PM{hour-12:02d}.{minute:02d}.{second:02d}"
    
    output_dir = SCRIPT_DIR / "outputs" / date_str / time_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def convert_vtt_to_timestamped_txt(vtt_path, output_path):
    """Convert VTT subtitle file to readable timestamped text format"""
    try:
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            lines = vtt_file.readlines()
        
        timestamped_content = []
        current_time = ""
        current_text = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and VTT header
            if not line or line == "WEBVTT":
                continue
            
            # Check if line contains timestamp (format: 00:01:23.456 --> 00:01:26.789)
            if " --> " in line:
                # If we have previous content, save it
                if current_time and current_text:
                    timestamped_content.append(f"[{current_time}] {current_text}")
                
                # Extract start time
                current_time = line.split(" --> ")[0]
                current_text = ""
            else:
                # This is text content
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
        
        # Don't forget the last entry
        if current_time and current_text:
            timestamped_content.append(f"[{current_time}] {current_text}")
        
        # Write the timestamped content
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write("# Timestamped Transcription\n\n")
            for entry in timestamped_content:
                output_file.write(entry + "\n\n")
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not create timestamped transcript: {e}")
        return False

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes is None:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_duration(duration_seconds):
    """Convert seconds to human readable format"""
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes}m {seconds}s"

def create_session_metadata(output_dir, input_file, args, ai_provider, ai_model, processing_start_time):
    """Create a metadata file with session details and processing options"""
    end_time = datetime.now()
    duration = (end_time - processing_start_time).total_seconds()
    file_size = input_file.stat().st_size if input_file.exists() else None
    
    metadata = {
        "session_info": {
            "start_time": processing_start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": format_duration(duration),
            "session_folder": str(output_dir.relative_to(SCRIPT_DIR))
        },
        "input_file": {
            "filename": input_file.name,
            "file_size": format_file_size(file_size)
        },
        "processing_options": {
            "transcription_model": args.transcription_model,
            "summarization_enabled": True,
            "summarization_model": ai_model,
            "summarization_provider": ai_provider,
            "summary_style": args.style,
            "output_format": args.format
        },
        "output_files": {
            "wav_file": f"{input_file.stem}.wav",
            "transcript_file": f"{input_file.stem}_transcript.txt",
            "summary_file": f"{input_file.stem}_summary.{args.format}",
            "vtt_file": f"{input_file.stem}_transcript.vtt",
            "metadata_file": "session_metadata.json"
        }
    }
    
    metadata_file = output_dir / "session_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_file

def is_youtube_url(url):
    """Check if the input string is a YouTube URL"""
    youtube_patterns = [
        r'https?://(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(www\.)?youtube\.com/embed/[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://m\.youtube\.com/watch\?v=[\w-]+',
        r'https?://youtube\.com/watch\?v=[\w-]+'
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

def sanitize_filename(filename):
    """Sanitize filename for safe filesystem storage"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:100]  # Limit length

def download_youtube_audio(url):
    """Download YouTube video as audio using yt-dlp"""
    print(f"🔗 Downloading YouTube audio from: {url}")
    
    # Check if yt-dlp is available
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: yt-dlp not found. Please install it:")
        print("   pip install yt-dlp")
        sys.exit(1)
    
    # Create dated inputs directory for today
    today = datetime.today().strftime('%Y-%m-%d')
    inputs_dir = SCRIPT_DIR / "inputs" / today
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Download audio to dated inputs directory
    output_template = str(inputs_dir / "%(title)s-%(id)s.%(ext)s")
    
    try:
        # Download best audio quality
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            '--extract-audio',
            '--audio-format', 'm4a',
            '--audio-quality', '0',  # Best quality
            '-o', output_template,
            '--restrict-filenames',  # Safe filenames
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded file
        # yt-dlp outputs the filename in its output
        output_lines = result.stdout.strip().split('\n')
        downloaded_file = None
        
        for line in output_lines:
            if 'has already been downloaded' in line or 'Destination:' in line:
                # Extract filename from output
                parts = line.split()
                for part in parts:
                    if part.startswith(str(inputs_dir)) and (part.endswith('.m4a') or part.endswith('.mp3')):
                        downloaded_file = Path(part)
                        break
        
        # If we couldn't parse the output, try to find the most recent file
        if not downloaded_file or not downloaded_file.exists():
            audio_files = list(inputs_dir.glob('*.m4a')) + list(inputs_dir.glob('*.mp3'))
            if audio_files:
                downloaded_file = max(audio_files, key=lambda f: f.stat().st_mtime)
        
        if downloaded_file and downloaded_file.exists():
            print(f"✅ Successfully downloaded: {downloaded_file.name}")
            return downloaded_file
        else:
            print("❌ Error: Could not locate downloaded file")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading YouTube video: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        sys.exit(1)

def create_directories():
    """Create necessary base directories"""
    (SCRIPT_DIR / "inputs").mkdir(exist_ok=True)
    (SCRIPT_DIR / "outputs").mkdir(exist_ok=True)
    (SCRIPT_DIR / "models").mkdir(exist_ok=True)
    (SCRIPT_DIR / "models" / "whisper").mkdir(exist_ok=True)
    (SCRIPT_DIR / "models" / "ollama").mkdir(exist_ok=True)

def organize_input_file(input_file):
    """Organize input file into dated folder if it's directly in inputs/ directory"""
    input_path = Path(input_file)
    inputs_dir = SCRIPT_DIR / "inputs"
    
    # Check if file is directly in inputs/ directory (not in a subfolder)
    if input_path.parent == inputs_dir:
        # Create today's dated folder
        today = datetime.today().strftime('%Y-%m-%d')
        dated_dir = inputs_dir / today
        dated_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file to dated folder
        organized_path = dated_dir / input_path.name
        if not organized_path.exists():
            shutil.move(str(input_path), str(organized_path))
            print(f"📁 Organized file into dated folder: {organized_path.relative_to(SCRIPT_DIR)}")
        
        return organized_path
    
    # File is already organized or not in inputs/ directory
    return input_path

def copy_input_file(input_file):
    """Copy input file to inputs directory if not already there"""
    audio_inputs_dir = SCRIPT_DIR / "inputs"
    target_path = audio_inputs_dir / Path(input_file).name
    
    # Only copy if the input file is not already the target file
    if Path(input_file).resolve() != target_path.resolve():
        shutil.copy2(input_file, target_path)
        print("Copied input file to inputs/")
    
    return target_path

def convert_to_wav(input_file, extension, basename, output_dir):
    """Convert input file to WAV format"""
    wav_file_path = output_dir / f"{basename}.wav"
    
    if extension == 'm4a':
        print("Converting M4A to WAV...")
        subprocess.run([
            sys.executable, 
            str(SCRIPT_DIR / "src" / "convert_audio.py"),
            str(input_file),
            str(wav_file_path)
        ], check=True)
    elif extension in ['mp4', 'mov']:
        print("Extracting audio from video and converting to WAV...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "src" / "extract_audio.py"),
            str(input_file),
            str(wav_file_path)
        ], check=True)
    elif extension == 'wav':
        print("Input is already WAV format, copying...")
        shutil.copy2(input_file, wav_file_path)
    elif extension == 'mp3':
        print("Converting MP3 to WAV...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "src" / "convert_audio.py"),
            str(input_file),
            str(wav_file_path)
        ], check=True)
    
    if not wav_file_path.exists():
        print(f"Error: WAV file was not created at {wav_file_path}")
        sys.exit(1)
    
    return wav_file_path

def download_whisper_model(model_name):
    """Download Whisper model if not exists locally"""
    model_file = SCRIPT_DIR / "models" / "whisper" / f"ggml-{model_name}.bin"
    
    if model_file.exists():
        return model_file
    
    print(f"\n📥 Downloading Whisper model: {model_name}")
    print("   This may take a few minutes...")
    
    # Download from Hugging Face
    model_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
        print(f"✅ Successfully downloaded {model_name}")
        return model_file
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        # Fallback to external path
        external_model = Path("/Users/eric/Desktop/2-Career/Projects/models/whisper") / f"ggml-{model_name}.bin"
        if external_model.exists():
            print(f"✅ Using external model at {external_model}")
            return external_model
        else:
            print("❌ Model not found. Please download manually.")
            sys.exit(1)

def transcribe_audio(wav_file_path, whisper_path, model_name, basename, output_dir):
    """Transcribe audio using Whisper"""
    print(f"\nTranscribing audio using Whisper model: {model_name}")
    
    # Download model if needed
    model_file = download_whisper_model(model_name)
    
    # Clean basename to avoid .wav.txt files
    clean_basename = basename.replace('.wav', '') if basename.endswith('.wav') else basename
    
    # Run whisper transcription with both text and timestamped output
    cmd = [
        str(whisper_path / "build" / "bin" / "whisper-cli"),
        "-m", str(model_file),
        "-f", str(wav_file_path),
        "-of", str(output_dir / clean_basename),
        "--output-txt",
        "--output-vtt",  # Add VTT output for timestamps
        "-t", "10",      # Use 10 threads for M2 Max
        "--flash-attn",  # Enable flash attention
        "-bs", "1",      # Reduce beam size for speed
        "-bo", "1",      # Reduce best-of candidates to match beam size
        "-mc", "4096",   # Increase max context (plenty of RAM available)
        "-nf"            # No temperature fallback for consistent speed
    ]
    
    subprocess.run(cmd, check=True)
    
    # Whisper creates .txt file, rename it to _transcript.txt for clarity
    whisper_txt_path = output_dir / f"{clean_basename}.txt"
    txt_output_path = output_dir / f"{clean_basename}_transcript.txt"
    
    # VTT file for timestamps (whisper creates .vtt, rename to _transcript.vtt)
    whisper_vtt_path = output_dir / f"{clean_basename}.vtt"
    vtt_output_path = output_dir / f"{clean_basename}_transcript.vtt"
    
    if not whisper_txt_path.exists():
        print("Error: Transcription failed - no output file created")
        sys.exit(1)
    
    # Rename the whisper outputs to include _transcript suffix
    whisper_txt_path.rename(txt_output_path)
    if whisper_vtt_path.exists():
        whisper_vtt_path.rename(vtt_output_path)
    
    print("\n✅ Transcription completed!")
    relative_path = txt_output_path.relative_to(SCRIPT_DIR)
    print(f"📄 Transcript saved to: {relative_path}")
    
    if vtt_output_path.exists():
        relative_vtt_path = vtt_output_path.relative_to(SCRIPT_DIR)
        print(f"⏰ Timestamps saved to: {relative_vtt_path}")
    
    return txt_output_path, clean_basename

def generate_summary(txt_output_path, clean_basename, summary_style, output_format, ai_provider, ai_model, output_dir, confidence_threshold=0.5, show_classification=False, interactive=False, context_hints=None):
    """Generate AI summary of the transcript"""
    if summary_style == "smart":
        print(f"\n🧠 Generating intelligent multi-category analysis...")
    else:
        print(f"\n🤖 Generating {summary_style} summary...")
    
    # Determine output file extension and path
    if summary_style == "smart":
        if output_format == "txt":
            summary_output_path = output_dir / f"{clean_basename}_smart_summary.txt"
        else:
            summary_output_path = output_dir / f"{clean_basename}_smart_summary.{output_format}"
    else:
        if output_format == "txt":
            summary_output_path = output_dir / f"{clean_basename}_summary.txt"
        else:
            summary_output_path = output_dir / f"{clean_basename}_summary.{output_format}"
    
    # Build summarization command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "src" / "summarize_transcript.py"),
        str(txt_output_path),
        "--style", summary_style,
        "--format", output_format,
        "--output", str(summary_output_path),
        "--provider", ai_provider
    ]
    
    if ai_model:
        cmd.extend(["--model", ai_model])
    
    if summary_style == "smart":
        cmd.extend(["--confidence-threshold", str(confidence_threshold)])
        if show_classification:
            cmd.append("--show-classification")
        if interactive:
            cmd.append("--interactive")
        if context_hints:
            cmd.extend(["--context-hints", context_hints])
    
    try:
        subprocess.run(cmd, check=True)
        if summary_style == "smart":
            print("✅ Multi-category analysis completed!")
        else:
            print("✅ Summary completed!")
        relative_path = summary_output_path.relative_to(SCRIPT_DIR)
        print(f"📋 Summary saved to: {relative_path}")
        return summary_output_path
    except subprocess.CalledProcessError:
        print("❌ Error: Summary generation failed")
        sys.exit(1)

def check_openai_api_key():
    """Check if OpenAI or Azure OpenAI credentials are available in .env file"""
    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        return False
    
    # Load .env file
    load_dotenv(env_file)
    
    # Check for Azure OpenAI credentials first
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    
    if (azure_api_key and azure_endpoint and azure_api_version and azure_deployment and
        azure_api_key.strip() != '' and azure_endpoint.strip() != '' and 
        azure_api_version.strip() != '' and azure_deployment.strip() != ''):
        return True
    
    # Fall back to regular OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key is not None and api_key.strip() != ''

def check_ollama_available():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return []


def parse_summarization_model(model_name):
    """Parse summarization model and determine provider"""
    openai_models = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
    ollama_models = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b', 'llama3.1:70b', 
                     'qwen2.5:7b', 'mistral:7b', 'codellama:7b']
    
    if model_name in openai_models:
        return "openai", model_name
    elif any(model_name.startswith(ollama_model.split(':')[0]) for ollama_model in ollama_models) or ':' in model_name:
        return "ollama", model_name
    else:
        # Try to auto-detect based on common patterns
        if model_name.startswith(('gpt-', 'text-')):
            return "openai", model_name
        elif ':' in model_name or model_name in ['llama', 'mistral', 'qwen', 'codellama']:
            return "ollama", model_name
        else:
            print(f"⚠️  Unknown model: {model_name}. Assuming OpenAI.")
            return "openai", model_name

def ensure_ollama_model(model_name):
    """Ensure Ollama model is available, download if needed"""
    if not check_ollama_available():
        print("❌ Ollama not available. Installing Ollama...")
        print("   Please install from: https://ollama.ai")
        sys.exit(1)
    
    models = get_ollama_models()
    if model_name not in models:
        print(f"\n📥 Downloading Ollama model: {model_name}")
        print("   This may take several minutes...")
        
        try:
            result = subprocess.run(['ollama', 'pull', model_name], timeout=1800)  # 30 min timeout
            if result.returncode == 0:
                print(f"✅ Successfully downloaded {model_name}")
            else:
                print(f"❌ Failed to download {model_name}")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print("❌ Download timed out. Please try again later.")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
            sys.exit(1)

def determine_ai_provider(model_name=None):
    """Smart AI provider detection with model-based fallback"""
    print("\n🤖 Determining AI provider and model...")
    
    if model_name:
        # User specified a model, determine provider from model name
        provider, model = parse_summarization_model(model_name)
        
        if provider == "openai":
            if check_openai_api_key():
                # Check which provider is actually available
                load_dotenv(SCRIPT_DIR / ".env")
                azure_creds = (os.getenv('AZURE_OPENAI_API_KEY') and 
                              os.getenv('AZURE_OPENAI_ENDPOINT') and 
                              os.getenv('AZURE_OPENAI_API_VERSION') and 
                              os.getenv('AZURE_OPENAI_DEPLOYMENT'))
                
                if azure_creds:
                    print(f"✅ Using Azure OpenAI with deployment for model: {model}")
                else:
                    print(f"✅ Using OpenAI with model: {model}")
                return provider, model
            else:
                print("⚠️  No OpenAI or Azure OpenAI credentials found in .env file")
                print(f"   Cannot use OpenAI model: {model}")
                print("   Falling back to local Ollama...")
                # Fallback to default Ollama model
                ensure_ollama_model("llama3.2:1b")
                return "ollama", "llama3.2:1b"
        
        elif provider == "ollama":
            ensure_ollama_model(model)
            print(f"✅ Using Ollama with model: {model}")
            return provider, model
    
    # No model specified, try default OpenAI first
    if check_openai_api_key():
        # Check which provider is actually available
        load_dotenv(SCRIPT_DIR / ".env")
        azure_creds = (os.getenv('AZURE_OPENAI_API_KEY') and 
                      os.getenv('AZURE_OPENAI_ENDPOINT') and 
                      os.getenv('AZURE_OPENAI_API_VERSION') and 
                      os.getenv('AZURE_OPENAI_DEPLOYMENT'))
        
        if azure_creds:
            print("✅ Using Azure OpenAI with default deployment")
        else:
            print("✅ Using OpenAI with default model: gpt-4o")
        return "openai", "gpt-4o"
    
    print("⚠️  No OpenAI or Azure OpenAI credentials found in .env file")
    print("   Falling back to local Ollama...")
    
    # Fallback to Ollama with default small model
    ensure_ollama_model("llama3.2:1b")
    print("✅ Using Ollama with default model: llama3.2:1b")
    return "ollama", "llama3.2:1b"

def process_single_file(input_file, args, whisper_path, ai_provider, ai_model):
    """Process a single audio/video file"""
    # Get file info
    input_filename = input_file.name
    input_extension = validate_file_extension(input_filename)
    input_basename = input_file.stem
    
    print(f"\n🎵 Processing {input_extension} file: {input_filename}")
    
    # Create timestamped output directory for this file
    output_dir = create_output_directory()
    processing_start_time = datetime.now()
    
    print(f"  Session output: {output_dir.relative_to(SCRIPT_DIR)}")
    
    # Copy input file to inputs if needed
    copy_input_file(input_file)
    
    # Step 1: Convert to WAV format
    wav_file_path = convert_to_wav(input_file, input_extension, input_basename, output_dir)
    
    # Step 2: Transcribe using Whisper
    txt_output_path, clean_basename = transcribe_audio(wav_file_path, whisper_path, args.transcription_model, input_basename, output_dir)
    
    # Step 3: Generate summary
    generate_summary(txt_output_path, clean_basename, args.style, args.format, ai_provider, ai_model, output_dir, 
                    args.confidence_threshold, args.show_classification, args.interactive, args.context_hints)
    
    # Create session metadata file
    metadata_file = create_session_metadata(output_dir, input_file, args, ai_provider, ai_model, processing_start_time)
    
    print(f"✅ Completed processing: {input_filename}")
    return output_dir

def find_audio_files(directory_path):
    """Find all supported audio/video files in a directory"""
    supported_extensions = ['.m4a', '.wav', '.mp3', '.mp4', '.mov']
    audio_files = []
    
    for ext in supported_extensions:
        audio_files.extend(directory_path.glob(f"**/*{ext}"))
    
    # Sort by name for consistent processing order
    return sorted(audio_files)

def detect_and_merge_segments(directory_path):
    """Detect and merge segment files in a directory before processing"""
    print(f"🔍 Checking for segment files in: {directory_path}")
    
    try:
        # Find all segment groups in the directory
        segment_groups = find_all_audio_video_files(str(directory_path))
        
        if not segment_groups:
            print("  ✅ No segment files detected - proceeding with individual files")
            return []
        
        print(f"  📋 Found {len(segment_groups)} file group(s) with segments:")
        for base_name, files in segment_groups.items():
            print(f"    • {base_name} ({len(files)} segments)")
        
        merged_files = []
        success_count = 0
        
        for base_name, files in segment_groups.items():
            print(f"\n  🔄 Merging: {base_name}")
            
            # Determine output file path and extension
            first_file = files[0]
            output_extension = first_file.suffix
            output_file = directory_path / f"{base_name}{output_extension}"
            
            # Merge the files
            if merge_files([str(f) for f in files], str(output_file)):
                print(f"    ✅ Successfully merged into: {output_file.name}")
                merged_files.append(output_file)
                success_count += 1
                
                # Organize segments into folder
                segments_dir = organize_segments(files, base_name)
                if segments_dir:
                    print(f"    📁 Segments organized in: {segments_dir.name}/")
            else:
                print(f"    ❌ Failed to merge {base_name}")
        
        if merged_files:
            print(f"\n  🎉 Successfully merged {success_count}/{len(segment_groups)} file groups")
            print(f"  📄 Created {len(merged_files)} merged files for processing")
        
        return merged_files
        
    except Exception as e:
        print(f"  ⚠️  Error during segment detection: {e}")
        print("  ✅ Proceeding with individual files")
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="TranscribeAndSummarizeAudioAndVideo - Unified transcription and summarization tool",
        add_help=False
    )
    
    parser.add_argument('input_file', nargs='?', help='Input audio/video file or directory')
    parser.add_argument('--transcription-model', default='medium.en', help='Whisper model for transcription (default: medium.en)')
    parser.add_argument('--summarization-model', default='', help='AI model for summarization (default: gpt-4o-mini)')
    parser.add_argument('--style', default='smart', help='Summary style: smart, general, meeting (default: smart)')
    parser.add_argument('--format', default='md', help='Output format: md, txt, pdf, json (default: md)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for smart analysis (default: 0.5)')
    parser.add_argument('--show-classification', action='store_true', help='Show classification results for smart analysis')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode for uncertain classifications')
    parser.add_argument('--context-hints', help='Provide context hints for ambiguous terms (format: term1=meaning1,term2=meaning2)')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message')
    
    args = parser.parse_args()
    
    # Show help if requested or no input file provided
    if args.help or not args.input_file:
        show_usage()
        sys.exit(0 if args.help else 1)
    
    # Check if input is a YouTube URL first
    if is_youtube_url(args.input_file):
        print(f"\n🔗 Detected YouTube URL: {args.input_file}")
        # Download the YouTube video as audio
        downloaded_file = download_youtube_audio(args.input_file)
        input_path = downloaded_file
    else:
        # Validate input path exists
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input path '{input_path}' not found")
            sys.exit(1)
        
        # Auto-organize files that are directly in inputs/ directory
        input_path = organize_input_file(input_path)
    
    # Determine if input is file or directory
    if input_path.is_file():
        # Single file processing (existing behavior)
        files_to_process = [input_path]
    elif input_path.is_dir():
        # Directory processing with intelligent segment detection and merging
        print(f"\n📁 Processing directory: {input_path}")
        
        # Step 1: Detect and merge segment files automatically
        merged_files = detect_and_merge_segments(input_path)
        
        # Step 2: Find all audio/video files (including newly merged ones)
        files_to_process = find_audio_files(input_path)
        if not files_to_process:
            print(f"Error: No supported audio/video files found in directory '{input_path}'")
            print("Supported formats: .m4a, .wav, .mp3, .mp4, .mov")
            sys.exit(1)
        
        # Show what will be processed
        if merged_files:
            print(f"\n📄 Found {len(files_to_process)} total files for processing (including {len(merged_files)} merged files)")
        else:
            print(f"\n📄 Found {len(files_to_process)} files for processing")
    else:
        print(f"Error: Input path '{input_path}' is neither a file nor a directory")
        sys.exit(1)
    
    # Validate arguments
    validate_summary_style(args.style)
    validate_output_format(args.format)
    
    # Find whisper path
    whisper_path = find_whisper_path()
    
    # Determine best AI provider with smart fallback
    ai_provider, ai_model = determine_ai_provider(args.summarization_model or None)
    
    # Show configuration
    if len(files_to_process) == 1:
        input_filename = files_to_process[0].name
        input_extension = validate_file_extension(input_filename)
        print(f"Processing {input_extension} file: {input_filename}")
    else:
        print(f"\n📁 Processing directory: {input_path}")
        print(f"Found {len(files_to_process)} audio/video files to process")
        for i, file in enumerate(files_to_process, 1):
            print(f"  {i}. {file.name}")
    
    print("\n📋 Configuration:")
    
    # Show transcription model with default indicator
    transcription_default = " (default)" if args.transcription_model == "medium.en" else ""
    print(f"  Transcription model: {args.transcription_model}{transcription_default}")
    
    # Show summarization model with default indicator
    summarization_default = " (default)" if ai_model == "gpt-4o" and not args.summarization_model else ""
    print(f"  Summarization model: {ai_model} ({ai_provider}){summarization_default}")
    
    # Show style with default indicator
    style_default = " (default)" if args.style == "general" else ""
    print(f"  Summary style: {args.style}{style_default}")
    
    # Show format with default indicator
    format_default = " (default)" if args.format == "md" else ""
    print(f"  Output format: {args.format}{format_default}")
    
    print(f"  Model weights storage: {SCRIPT_DIR}/models/")
    print()
    
    # Create base directories
    create_directories()
    
    # Process all files
    completed_sessions = []
    for i, file in enumerate(files_to_process, 1):
        if len(files_to_process) > 1:
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(files_to_process)}")
            print(f"{'='*60}")
        
        try:
            session_dir = process_single_file(file, args, whisper_path, ai_provider, ai_model)
            completed_sessions.append((file.name, session_dir))
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")
            continue
    
    # Show completion summary
    if len(files_to_process) == 1:
        print("\n🎉 Processing completed successfully!")
        if completed_sessions:
            file_name, session_dir = completed_sessions[0]
            clean_basename = Path(file_name).stem
            print("\nOutput files in session folder:")
            print(f"  📄 Transcript: {clean_basename}_transcript.txt")
            print(f"  ⏰ Timestamps: {clean_basename}_transcript.vtt")
            print(f"  📋 Summary: {clean_basename}_summary.{args.format}")
            print(f"  🎵 Audio: {clean_basename}.wav")
            print(f"  📊 Metadata: session_metadata.json")
            print(f"\nSession folder: {session_dir.relative_to(SCRIPT_DIR)}")
    else:
        print(f"\n🎉 Batch processing completed!")
        print(f"Successfully processed {len(completed_sessions)}/{len(files_to_process)} files")
        print("\nSession folders created:")
        for file_name, session_dir in completed_sessions:
            print(f"  📁 {file_name} → {session_dir.relative_to(SCRIPT_DIR)}")
    print()

if __name__ == "__main__":
    main()