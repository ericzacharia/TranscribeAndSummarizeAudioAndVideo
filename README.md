# 🎵 TranscribeAndSummarizeAudioAndVideo

An intelligent tool that automatically transcribes and summarizes audio/video files using local Whisper.cpp for transcription and AI-powered multi-category analysis for summarization. Features smart content classification that adapts summaries based on content type (meetings, learning content, technical discussions, etc.).

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

# YouTube URL processing
python run.py "https://www.youtube.com/watch?v=VIDEO_ID"
python run.py "https://youtu.be/VIDEO_ID" --style meeting --format pdf

# Directory batch processing (process all audio/video files)
python run.py inputs/2025-07-15/
python run.py inputs/ --style meeting --format pdf

# Smart multi-category analysis (default)
python run.py input_video.mp4                                    # Intelligent classification & tailored summaries
python run.py input_file.m4a --show-classification               # Show confidence scores for each category

# Interactive mode for uncertain classifications
python run.py input_meeting.m4a --interactive                    # Get prompted when classification is uncertain

# Provide context hints for domain-specific terms
python run.py input_technical.m4a --context-hints "entities=data_objects,opsec=security,ui=interface"

# Traditional summary styles
python run.py input_video.mp4 --style meeting --format md        # Classic meeting format
python run.py input_video.mp4 --style general --format pdf       # Classic general format

# Use local Ollama instead of OpenAI
python run.py input_file.m4a --summarization-model llama3.2:1b

# Options: --transcription-model [tiny|base|small|medium|medium.en|large-v3], 
#          --summarization-model [gpt-4o|gpt-4o-mini|llama3.2:1b|llama3.2:3b|llama3.1:8b], 
#          --style [smart|general|meeting], --format [txt|md|pdf|json]
#          --confidence-threshold [0.0-1.0], --interactive, --context-hints
```

### Streamlit Web App
```bash
streamlit run src/app.py
```

## Features

### 🧠 **Intelligent Content Classification**
- **Smart Analysis**: Automatically detects content type (technical meetings, learning content, status updates, etc.)
- **Multi-Category Summaries**: Generates tailored summaries for multiple detected categories
- **Confidence Scoring**: Shows classification confidence with adjustable thresholds
- **12 Content Categories**: Professional (technical meetings, project planning, research, etc.) + Personal (reflection, life planning, social, learning)

### 🎯 **Adaptive Summarization**
- **Category-Specific Templates**: Each content type gets optimized summary structure
- **Unified Action Dashboard**: Consolidates action items across all categories
- **Work-Ready Outputs**: JIRA tickets, technical documentation, learning plans, etc.
- **Interactive Mode**: User input for uncertain classifications
- **Context Hints**: Provide meaning for domain-specific terminology

### 📁 **Core Functionality**
- **Input formats**: M4A, MP4, MOV, WAV files + **YouTube URLs**
- **Transcription**: Local Whisper.cpp (tiny to large models)
- **AI Options**: OpenAI GPT (default) or local Ollama/Llama models
- **Output formats**: TXT, MD, PDF, JSON
- **Dual interfaces**: CLI and web app
- **Auto-organization**: Files automatically organized into dated folders
- **Segment Detection**: Automatic merging of multi-part recordings

## File Organization

The system automatically organizes all input files into dated folders for better organization:

- **YouTube downloads**: Saved to `inputs/YYYY-MM-DD/Video_Title-ID.m4a`
- **Manual file processing**: Files in `inputs/` root are moved to `inputs/YYYY-MM-DD/filename`
- **Session outputs**: Always saved to `outputs/YYYY-MM-DD/[AM|PM]HH.MM.SS/`

**Example organization behavior:**
```bash
# Before processing
inputs/
├── my_video.mp4          # File directly in inputs/
└── 2025-07-27/
    └── older_file.m4a    # Already organized

# After: python run.py inputs/my_video.mp4
inputs/
├── 2025-07-28/
│   └── my_video.mp4      # Auto-moved to today's folder
├── 2025-07-27/
│   └── older_file.m4a    # Unchanged (already organized)
```

**Benefits:**
- ✅ Clean organization by date
- ✅ Easy to find recent files
- ✅ Consistent with output structure
- ✅ Non-destructive (only moves root-level files)

## Examples

```bash
# Smart Analysis (New Default)
python run.py "Meeting Notes.m4a"                                    # Intelligent multi-category analysis
python run.py "Team_Meeting.mp4" --show-classification               # Show what categories were detected
python run.py "Technical_Discussion.m4a" --interactive               # Interactive mode for uncertain content

# Context-Aware Processing
python run.py "OPSEC_Meeting.m4a" --context-hints "entities=data_objects,opsec=security_analysis"
python run.py "Code_Review.mp4" --context-hints "api=interface,nlp=language_processing" --format json

# Traditional Styles (Still Available)
python run.py "Lecture.mp4" --style general --transcription-model large-v3   # Classic general format
python run.py "Team_Meeting.mp4" --style meeting --format pdf               # Classic meeting format

# YouTube URL processing
python run.py "https://www.youtube.com/watch?v=QT6T6AC02-Q"          # Smart analysis of YouTube content
python run.py "https://youtu.be/ABC123" --interactive                # Interactive YouTube processing

# Advanced Configuration
python run.py "Workshop.mp4" --confidence-threshold 0.3 --show-classification  # Lower threshold, show all scores
python run.py "Research_Call.m4a" --summarization-model gpt-4o --format json   # Structured JSON output

# Directory batch processing with smart analysis
python run.py inputs/2025-07-15/                                     # Smart analysis for all files
python run.py recordings/ --interactive --format md                  # Interactive batch processing
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
- **"yt-dlp not found"**: Install yt-dlp with `pip install --user --break-system-packages yt-dlp`
- **YouTube download fails**: Check internet connection and video availability
- **FFmpeg errors**: Install FFmpeg with `sudo apt install ffmpeg`
- **Large files timeout**: Use smaller transcription models (`--transcription-model tiny`) for faster processing
- **Memory issues**: Close other applications or use smaller models

### Smart Analysis Specific
- **Wrong category detected**: Use `--interactive` mode to correct and teach the system
- **Technical terms misunderstood**: Use `--context-hints` to provide domain-specific meanings
- **Low confidence scores**: Try lowering `--confidence-threshold` or using `--interactive` mode
- **Multiple categories needed**: The system automatically generates summaries for all categories above threshold

### WSL/Linux Specific
- **Systemd warnings**: Normal in WSL, Ollama still works with manual start
- **Permission errors**: Use `--user` flag with pip or check file permissions
- **Audio conversion fails**: Ensure FFmpeg is installed and accessible

## License

MIT License - see [LICENSE](LICENSE) file for details

---

⭐ **Star this repository if you find it useful!**