# 🎵 TranscribeAndSummarizeAudioAndVideo

A tool that automatically transcribes and summarizes audio/video files using local Whisper.cpp for transcription and OpenAI GPT (default) or local Ollama/Llama models for summarization.

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone git@github.com:ericzacharia/TranscribeAndSummarizeAudioAndVideo.git
cd TranscribeAndSummarizeAudioAndVideo

# Option 1: Conda (Recommended)
conda env create -f setup/environment.yml
conda activate TranscribeAndSummarizeAudioAndVideo

# Option 2: pip
python -m venv venv && source venv/bin/activate
pip install -r setup/requirements.txt

# Setup whisper.cpp and configure
python setup/setup.py

# Configure OpenAI (default)
cp .env.template .env
# Edit .env with your OpenAI API key

# OR install Ollama for local AI (alternative)
ollama pull llama3.1
```

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

# Directory batch processing
python run.py inputs/2025-07-15/                                     # Process all files in date folder
python run.py recordings/ --style meeting --format pdf               # Meeting transcripts as PDFs
python run.py interviews/ --transcription-model large-v3 --format md # High quality batch processing
```

## Utilities

### Audio/Video File Merging
Use `merge_media.py` to merge multiple audio/video recordings (requires ffmpeg):

**Features:**
- Merges multiple files (not just 2)
- Auto-detects segment files by base name
- Supports audio (.m4a, .wav, .mp3) and video (.mp4, .mov) files
- Intelligent file naming (removes segment indicators like (1-2), (2-2))
- Organizes original segments into `<basename>_segments/` folder
- Clean output filenames without segment indicators

```bash
# Install ffmpeg first (one-time setup)
sudo apt install ffmpeg

# Auto-detect and merge segments by base name
python merge_media.py "Hunt Working Session 7.15"
python merge_media.py "Meeting Recording" -d inputs/2025-07-15/

# Manual file specification
python merge_media.py file1.m4a file2.m4a file3.m4a
python merge_media.py file1.mp4 file2.mp4 -o merged_video.mp4

# Dry run to see what would be merged
python merge_media.py "Recording" --dry-run
```

**Example workflow:**
```bash
# Before: Hunt Working Session 7.15 (1-2).m4a, Hunt Working Session 7.15 (2-2).m4a
python merge_media.py "Hunt Working Session 7.15"
# After: Hunt Working Session 7.15.m4a + Hunt Working Session 7.15_segments/ folder
```

## Troubleshooting

- **Missing OpenAI key**: Configure OpenAI key in `.env` or use `--model ollama`
- **Ollama not running**: Start with `ollama serve` or use default OpenAI
- **Whisper model not found**: Run `python setup/setup.py` to download models
- **FFmpeg missing**: Install FFmpeg for video processing
- **Large files**: Use smaller models (tiny/base) for faster processing

## License

MIT License - see [LICENSE](LICENSE) file for details

---

⭐ **Star this repository if you find it useful!**