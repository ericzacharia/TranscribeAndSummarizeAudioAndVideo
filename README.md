# 🎵 TranscribeAndSummarizeAudioAndVideo

A tool that automatically transcribes and summarizes audio/video files using local Whisper.cpp for transcription and OpenAI GPT (default) or local Ollama/Llama models for summarization.

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone git@github.com:ericzacharia/TranscribeAndSummarizeAudioAndVideo.git
cd TranscribeAndSummarizeAudioAndVideo

# Install dependencies (Linux/WSL)
pip install --user --break-system-packages -r setup/requirements.txt

# Setup whisper.cpp and configure
python setup/setup.py

# Configure OpenAI (default)
cp .env.template .env
# Edit .env with your OpenAI API key

# OR install Ollama for local AI (alternative)
curl -fsSL https://ollama.com/install.sh | sh
```

### Ollama Setup for Local AI (WSL/Linux)
If using local Ollama models instead of OpenAI:

```bash
# Start Ollama server (required before each use)
ollama serve &

# Pull your desired model (one-time setup)
ollama pull llama3.2:3b

# Now you can use local models
python run.py input.m4a --summarization-model llama3.2:3b
```

**Ollama Management Tips:**
- **Start when needed**: `ollama serve &` (runs in background)
- **Check if running**: `ps aux | grep ollama`
- **Stop when done**: `pkill ollama`
- **Convenient alias**: Add to `.bashrc`: `alias start-transcribe="ollama serve & sleep 2 && echo 'Ollama ready'"`

**Note**: In WSL, Ollama doesn't auto-start. You'll need to manually start it before using local models.

## Usage

### Command Line
```bash
# Single file processing
python run.py input_file.m4a
python run.py input_video.mp4

# Directory batch processing (process all audio/video files)
python run.py inputs/2025-07-15/
python run.py inputs/ --style meeting --format pdf

# Custom summary styles and formats
python run.py input_video.mp4 --style meeting --format md

# Use local Ollama instead of OpenAI
python run.py input_file.m4a --summarization-model llama3.2:1b

# Options: --transcription-model [tiny|base|small|medium|medium.en|large-v3], 
#          --summarization-model [gpt-4o|gpt-4o-mini|llama3.2:1b|llama3.2:3b|llama3.1:8b], 
#          --style [general|meeting], --format [txt|md|pdf]
```

### Streamlit Web App
```bash
streamlit run src/app.py
```

## Features

- **Formats**: M4A, MP4, MOV, WAV files
- **Transcription**: Local Whisper.cpp (tiny to large models)
- **Summarization**: OpenAI GPT (default) or local Ollama/Llama models
- **Summary styles**: General and meeting formats
- **Output**: TXT, MD, PDF formats
- **Dual interfaces**: CLI and web app

## Examples

```bash
# Single file processing
python run.py "Meeting Notes.m4a"                                    # iPhone voice memo with defaults
python run.py "Team_Meeting.mp4" --style meeting                     # Zoom recording with meeting format
python run.py "Lecture.mp4" --transcription-model large-v3           # High quality transcription
python run.py "Audio File.m4a" --summarization-model gpt-4o          # Custom OpenAI model

# Directory batch processing with automatic segment merging
python run.py inputs/2025-07-15/                                     # Auto-detects and merges segments, then processes all files
python run.py recordings/ --style meeting --format pdf               # Meeting transcripts as PDFs with automatic merging
python run.py interviews/ --transcription-model large-v3 --format md # High quality batch processing with smart segment handling
```

## Intelligent Segment Detection and Merging

The system automatically detects and merges segment files when processing directories, eliminating the need for manual merging:

**How it works:**
- When you point to a directory, the system first scans for segment files
- Automatically detects patterns like `(1-2)`, `(2-2)`, `_part1`, `_part2`, etc.
- Merges segments into clean files with organized storage
- Proceeds with transcription of merged and individual files

**Example workflow:**
```bash
# Directory contains: "Hunt Session 7.15 (1-2).m4a", "Hunt Session 7.15 (2-2).m4a", "meeting.mp4"
python run.py inputs/2025-07-15/

# System automatically:
# 1. ✅ Detects: "Hunt Session 7.15" has 2 segments  
# 2. 🔄 Merges: Creates "Hunt Session 7.15.m4a"
# 3. 📁 Organizes: Moves segments to "Hunt Session 7.15_segments/" folder
# 4. 📄 Processes: Both merged file and "meeting.mp4" for transcription
```

**Benefits:**
- **Seamless workflow**: No manual merging required
- **Smart detection**: Handles various segment naming patterns
- **Clean organization**: Original segments safely stored in subfolders
- **Flexible**: Works with any directory structure

## Utilities

### Manual Audio/Video File Merging
Use `src/merge_media.py` for standalone merging (requires ffmpeg):

**Features:**
- Auto-detects if input is directory or file/base name
- Merges multiple files with intelligent naming
- Supports audio (.m4a, .wav, .mp3) and video (.mp4, .mov) files
- Organizes original segments into `<basename>_segments/` folder
- Optimized for inputs/date/ folder structure

```bash
# Install ffmpeg first (one-time setup)
sudo apt install ffmpeg

# Auto-detect directory and merge all segment files
python src/merge_media.py inputs/2025-07-15/

# Auto-detect segments by base name
python src/merge_media.py "Hunt Working Session 7.15"

# Manual file specification
python src/merge_media.py file1.m4a file2.m4a file3.m4a

# Dry run to see what would be merged
python src/merge_media.py inputs/2025-07-15/ --dry-run
```

**Note:** Manual merging is rarely needed since `run.py` automatically handles segment detection and merging.

## Troubleshooting

### Common Issues
- **Missing OpenAI key**: Configure OpenAI key in `.env` or use local Ollama models
- **"Ollama not available"**: Start Ollama with `ollama serve &` before running with `--summarization-model llama3.2:3b`
- **"whisper.cpp not found"**: Run `python setup/setup.py` or manually build whisper.cpp
- **"No module named 'pydub'"**: Install dependencies with `pip install --user --break-system-packages -r setup/requirements.txt`
- **FFmpeg errors**: Install FFmpeg with `sudo apt install ffmpeg`
- **Large files timeout**: Use smaller transcription models (`--transcription-model tiny`) for faster processing
- **Memory issues**: Close other applications or use smaller models

### WSL/Linux Specific
- **Systemd warnings**: Normal in WSL, Ollama still works with manual start
- **Permission errors**: Use `--user` flag with pip or check file permissions
- **Audio conversion fails**: Ensure FFmpeg is installed and accessible

## License

MIT License - see [LICENSE](LICENSE) file for details

---

⭐ **Star this repository if you find it useful!**