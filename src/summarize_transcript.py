#!/usr/bin/env python3
"""
Enhanced transcript summarization supporting multiple styles and output formats.
Supports both local Ollama models (default) and OpenAI models.
"""

import argparse
import os
import sys
import json
import requests
from dotenv import load_dotenv
from datetime import datetime
from fpdf import FPDF
import markdown2

# Load environment variables from the project root
from pathlib import Path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

def get_ollama_client():
    """Initialize Ollama client and check if service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

def get_openai_client():
    """Initialize OpenAI client if API key is available."""
    try:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return OpenAI(api_key=api_key)
    except ImportError:
        pass
    return None

# Determine which client to use
ollama_available = get_ollama_client()
openai_client = get_openai_client()


def call_ollama(prompt, model="llama3.1"):
    """Call Ollama API for text generation."""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to call Ollama: {str(e)}")

def call_openai(prompt, system_message, model="gpt-4o-mini"):
    """Call OpenAI API for text generation."""
    if not openai_client:
        raise Exception("OpenAI client not available. Check API key.")
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content.strip()

def summarize_general(transcript, use_openai=False, model=None):
    """
    Generate a general-purpose summary suitable for lectures, interviews, etc.
    """
    prompt = f"""Please create a comprehensive but concise summary of the following transcript.

Structure your summary with:

## Main Topics
- List the key topics discussed

## Key Insights
- Important ideas, concepts, or information shared
- Notable quotes or statements (if any)

## Key Points
- Main arguments or points made
- Supporting details or examples

## Action Items & Next Steps
- Any actionable items mentioned
- Follow-up tasks or recommendations
- Questions to explore further

## Conclusion
- Brief summary of the overall discussion
- Main takeaways

Keep the summary clear, well-organized, and focused on the most important information.

Transcript:
{transcript}"""
    
    if use_openai:
        return call_openai(prompt, "You are an expert at creating clear, structured summaries of transcripts.", model or "gpt-4o-mini")
    else:
        full_prompt = "You are an expert at creating clear, structured summaries of transcripts.\n\n" + prompt
        return call_ollama(full_prompt, model or "llama3.1")


def summarize_meeting(transcript, use_openai=False, model=None):
    """
    Generate a detailed meeting summary with action items and JIRA-style tickets.
    Based on the video transcription project's meeting format.
    """
    prompt = f"""You are an executive assistant. The following is a meeting transcript where participants discuss action items, projects, and tasks.
Summarize the transcript and extract clear, actionable information. Structure the summary into the following sections:

### 1. Comprehensive Meeting Summary
- Provide a high-level overview of the meeting discussion.
- Highlight key decisions made and the reasoning behind them.
- Identify any unresolved questions or pending clarifications.

### 2. Actionable Tasks List
- Organize tasks into **logical phases** based on the discussion (e.g., Planning, Development, Testing, Deployment, Data Analysis, etc.).
    - Do **not** hardcode specific phase names—determine them dynamically from the meeting content.
- **For each task, include:**
    - **Task Description**: What needs to be done and why.
    - **Steps to Completion**: Break the task into key sub-steps if necessary.
    - **Dependencies**: Mention any blockers or prerequisites.
    - **Responsible Team Members** (if mentioned).
    - **Deadlines or time estimates** (if specified).

### 3. Timeline
- Create a structured timeline for task completion.
- Assume the team works in **two-week sprints** and follows a **Monday-Friday work schedule**.
- Use today's date ({datetime.today().strftime('%Y-%m-%d')}) as a reference point to estimate due dates.
- If specific deadlines are mentioned in the transcript, incorporate them into the timeline.

### 4. Technical Implementation Notes (if applicable)
- Summarize any technical discussions, best practices, or guidelines shared during the meeting.
- Note any tools, frameworks, or coding standards that must be followed.

### 5. JIRA Task Creation
- Generate **JIRA-style** tickets for each actionable task identified in the meeting.
- Each ticket should include:
    - **Task Summary**: A concise title that clearly describes the work to be done.
    - **Description**:
        - Clearly explain the task, including relevant context or dependencies.
        - Reference any relevant documentation, field mappings, or data sources if applicable.
    - **Definition of Done**:
        - List measurable criteria that define task completion.
        - Include expected outputs (e.g., features implemented, tests written and passing, documentation updated).
    - **Priority Level** (if mentioned in the discussion).
    - **Sprint and Due Dates** (based on the structured timeline, assuming a **two-week sprint schedule, Monday-Friday**).
    - **Assignee (if specified)**.
    - **Parent Epic (if applicable)**.
    - **Story Points** (estimate effort required if discussed).
    - **Acceptance Criteria**: Define what constitutes task completion.

### 6. Follow-up Items
- Questions that need answers
- Information that needs to be gathered
- People who need to be contacted
- Decisions that need to be made

**Transcript:**
{transcript}"""
    
    if use_openai:
        return call_openai(prompt, "You are a professional meeting secretary with expertise in project management and task organization.", model or "gpt-4o-mini")
    else:
        full_prompt = "You are a professional meeting secretary with expertise in project management and task organization.\n\n" + prompt
        return call_ollama(full_prompt, model or "llama3.1")


def save_as_pdf(content, output_path):
    """Save content as PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Split content into lines and handle UTF-8 encoding
    lines = content.split('\n')
    for line in lines:
        # Handle special characters
        try:
            # Replace common problematic characters
            line = line.replace('•', '-').replace("'", "'").replace('"', '"').replace('"', '"')
            line = line.encode('latin1', 'ignore').decode('latin1')
            pdf.cell(0, 5, txt=line, ln=True)
        except:
            # Skip lines that can't be encoded
            pdf.cell(0, 5, txt="[Line contains unsupported characters]", ln=True)
    
    pdf.output(output_path)


def save_as_markdown(content, output_path):
    """Save content as Markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def save_as_text(content, output_path, style, transcript_file):
    """Save content as plain text file with header."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Summary of: {os.path.basename(transcript_file)}\n")
        f.write(f"Style: {style.title()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced transcript summarization with local Ollama (default) and OpenAI support",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("transcript_file", help="Path to the transcript file")
    parser.add_argument("-s", "--style", choices=["general", "meeting"], 
                       default="general", help="Summary style (default: general)")
    parser.add_argument("-f", "--format", choices=["txt", "md", "pdf"], 
                       default="txt", help="Output format (default: txt)")
    parser.add_argument("-o", "--output", help="Output file path (default: append _summary to input filename)")
    parser.add_argument("--provider", choices=["ollama", "openai"], 
                       default="ollama", help="AI provider (default: ollama)")
    parser.add_argument("--model", help="Specific model to use (default: llama3.1 for ollama, gpt-4o-mini for openai)")
    
    args = parser.parse_args()
    
    # Check provider availability
    if args.provider == "ollama" and not ollama_available:
        print("Error: Ollama is not running. Please start Ollama or use --provider openai")
        print("To start Ollama: ollama serve")
        sys.exit(1)
    
    if args.provider == "openai" and not openai_client:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY in .env file or use --provider ollama")
        sys.exit(1)
    
    use_openai = args.provider == "openai"
    
    # Read the transcript
    try:
        with open(args.transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
    except FileNotFoundError:
        print(f"Error: Transcript file '{args.transcript_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading transcript file: {str(e)}")
        sys.exit(1)
    
    if not transcript.strip():
        print("Error: Transcript is empty.")
        sys.exit(1)
    
    # Generate summary based on style
    provider_name = "OpenAI" if use_openai else "Ollama"
    model_name = args.model or ("gpt-4o-mini" if use_openai else "llama3.1")
    print(f"Generating {args.style} summary using {provider_name} ({model_name})...")
    
    try:
        if args.style == "meeting":
            summary = summarize_meeting(transcript, use_openai, args.model)
        else:
            summary = summarize_general(transcript, use_openai, args.model)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.transcript_file)[0]
        output_path = f"{base_name}_summary.{args.format}"
    
    # Save summary based on format
    try:
        if args.format == "pdf":
            save_as_pdf(summary, output_path)
        elif args.format == "md":
            save_as_markdown(summary, output_path)
        else:  # txt
            save_as_text(summary, output_path, args.style, args.transcript_file)
        
        print(f"Summary saved to: {output_path}")
    except Exception as e:
        print(f"Error writing summary file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()