# 🎵 TranscribeAndSummarizeAudioAndVideo

A tool that automatically transcribes and summarizes audio/video files using local Whisper.cpp for transcription and OpenAI GPT (default) or local Ollama/Llama models for summarization.

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone git@github.com:ericzacharia/TranscribeAndSummarizeAudioAndVideo.git
cd TranscribeAndSummarizeAudioAndVideo

# Option 1: Conda (Recommended)
conda env create -f environment.yml
conda activate TranscribeAndSummarizeAudioAndVideo

# Option 2: pip
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Setup whisper.cpp and configure
python setup.py

# Configure OpenAI (default)
cp .env.template .env
# Edit .env with your OpenAI API key

# OR install Ollama for local AI (alternative)
ollama pull llama3.1
```

## Usage

### Command Line
```bash
# Basic transcription and summarization (uses OpenAI by default)
python run.py input_file.m4a
python run.py input_video.mp4

# Custom summary styles and formats
python run.py input_video.mp4 --style meeting --format md

# Use local Ollama instead of OpenAI
python run.py input_file.m4a --model ollama

# Options: --style [general|meeting], --whisper-model [tiny|base|small|medium|large], 
#          --format [txt|md|pdf], --model [ollama|openai], --ai-model [model_name]
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
# iPhone voice memo (transcribes and summarizes with OpenAI by default)
python run.py "Meeting Notes.m4a"

# Zoom recording with meeting-style summary
python run.py "Team_Meeting.mp4" --style meeting

# Lecture with best quality using Ollama
python run.py "Lecture.mp4" --whisper-model large --model ollama --ai-model llama3.1

# Use custom OpenAI model
python run.py "Audio File.m4a" --ai-model gpt-4
```

## Troubleshooting

- **Missing OpenAI key**: Configure OpenAI key in `.env` or use `--model ollama`
- **Ollama not running**: Start with `ollama serve` or use default OpenAI
- **Whisper model not found**: Run `./setup.sh` to download models
- **FFmpeg missing**: Install FFmpeg for video processing
- **Large files**: Use smaller models (tiny/base) for faster processing

## License

MIT License - see [LICENSE](LICENSE) file for details

---

⭐ **Star this repository if you find it useful!**