#!/usr/bin/env python3
"""
Enhanced transcript summarization with intelligent multi-category classification.
Supports both local Ollama models (default) and OpenAI models.
Features 12-category content classification with confidence-based multi-output generation.
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
from typing import List, Dict, Any

# Load environment variables from the project root
from pathlib import Path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Import our new classification and template modules
from content_classifier import ContentClassifier, ContentCategory, MultiCategoryResult
from summary_templates import SummaryTemplateEngine
from classification_feedback import InteractiveClassifier, add_context_hints

def get_ollama_client():
    """Initialize Ollama client and check if service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

def get_azure_openai_credentials():
    """Check for Azure OpenAI credentials."""
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    
    if api_key and endpoint and api_version and deployment:
        return {
            'api_key': api_key,
            'endpoint': endpoint,
            'api_version': api_version,
            'deployment': deployment
        }
    return None

def get_openai_client():
    """Initialize OpenAI client, supporting both regular OpenAI and Azure OpenAI."""
    try:
        # First check for Azure OpenAI credentials
        azure_creds = get_azure_openai_credentials()
        if azure_creds:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_creds['api_key'],
                azure_endpoint=azure_creds['endpoint'],
                api_version=azure_creds['api_version']
            )
            # Store deployment name for later use
            client._deployment_name = azure_creds['deployment']
            client._is_azure = True
            return client
        
        # Fall back to regular OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client._is_azure = False
            return client
            
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
    """Call OpenAI API for text generation (supports both OpenAI and Azure OpenAI)."""
    if not openai_client:
        azure_creds = get_azure_openai_credentials()
        if azure_creds:
            raise Exception("Azure OpenAI credentials found but client initialization failed. Check configuration.")
        else:
            raise Exception("OpenAI client not available. Check API key in .env file (OPENAI_API_KEY or Azure OpenAI credentials).")
    
    # Use deployment name for Azure, model name for regular OpenAI
    if hasattr(openai_client, '_is_azure') and openai_client._is_azure:
        model_or_deployment = openai_client._deployment_name
    else:
        model_or_deployment = model
    
    response = openai_client.chat.completions.create(
        model=model_or_deployment,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content.strip()

def generate_multi_category_summary(transcript: str, use_openai: bool = False, model: str = None, 
                                   confidence_threshold: float = 0.5, classifier: ContentClassifier = None) -> Dict[str, Any]:
    """
    Generate intelligent multi-category summary with confidence-based output generation.
    
    Args:
        transcript: Raw transcript text to analyze
        use_openai: Whether to use OpenAI instead of Ollama
        model: Specific model to use
        confidence_threshold: Minimum confidence for category inclusion
        
    Returns:
        Dictionary containing classification results and category-specific summaries
    """
    # Initialize classifier and template engine
    if classifier is None:
        classifier = ContentClassifier(confidence_threshold=confidence_threshold)
    template_engine = SummaryTemplateEngine()
    
    # Classify the transcript
    classification_result = classifier.classify_content(transcript)
    
    # Generate summaries for all categories above threshold
    category_summaries = {}
    for category in classification_result.above_threshold_categories:
        # Get category-specific prompt
        prompt = template_engine.format_prompt(category, transcript)
        
        # Generate summary using appropriate AI service
        if use_openai:
            summary = call_openai(prompt, f"You are an expert at analyzing {category.value} content.", model)
        else:
            summary = call_ollama(prompt, model or "llama3.1")
        
        # Apply post-processing
        summary = template_engine.post_process_summary(category, summary)
        
        category_summaries[category.value] = {
            'summary': summary,
            'confidence': classification_result.get_confidence(category),
            'key_indicators': next((r.key_indicators for r in classification_result.results 
                                  if r.category == category), [])
        }
    
    # Generate unified action dashboard
    unified_actions = generate_unified_action_dashboard(category_summaries, classification_result)
    
    return {
        'classification': {
            'primary_category': classification_result.primary_category.value,
            'above_threshold_categories': [cat.value for cat in classification_result.above_threshold_categories],
            'all_confidence_scores': {result.category.value: result.confidence 
                                    for result in classification_result.results}
        },
        'category_summaries': category_summaries,
        'unified_actions': unified_actions,
        'metadata': {
            'confidence_threshold': confidence_threshold,
            'total_categories_analyzed': len(ContentCategory),
            'categories_above_threshold': len(classification_result.above_threshold_categories),
            'ai_provider': 'openai' if use_openai else 'ollama',
            'model': model or ('gpt-4o-mini' if use_openai else 'llama3.1'),
            'generated_at': datetime.now().isoformat()
        }
    }

def generate_unified_action_dashboard(category_summaries: Dict[str, Any], 
                                    classification_result: MultiCategoryResult) -> Dict[str, List[str]]:
    """
    Generate a unified action dashboard consolidating items across all categories.
    """
    actions = {
        'immediate_actions': [],
        'this_week': [],
        'this_month': [],
        'ongoing_goals': [],
        'follow_up_needed': []
    }
    
    # Extract action items from each category summary
    for category_name, summary_data in category_summaries.items():
        summary_text = summary_data['summary']
        
        # Extract different types of action items using pattern matching
        # Look for checkbox items with different time indicators
        import re
        
        # Immediate actions (next 24-48 hours, immediate)
        immediate_patterns = [
            r'- \[ \] ([^\n]*(?:immediate|24|48|hours|today|tomorrow)[^\n]*)',
            r'\*\*Immediate[^\*]*\*\*[^\n]*\n([^\n#]*(?:- \[ \])[^\n]*)',
        ]
        
        for pattern in immediate_patterns:
            matches = re.findall(pattern, summary_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.strip():
                    actions['immediate_actions'].append(f"[{category_name}] {match.strip()}")
        
        # This week actions
        week_patterns = [
            r'- \[ \] ([^\n]*(?:this week|week|weekly)[^\n]*)',
            r'\*\*.*[Ww]eek.*\*\*[^\n]*\n([^\n#]*(?:- \[ \])[^\n]*)',
        ]
        
        for pattern in week_patterns:
            matches = re.findall(pattern, summary_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.strip():
                    actions['this_week'].append(f"[{category_name}] {match.strip()}")
        
        # Follow-up actions
        followup_patterns = [
            r'- \[ \] ([^\n]*(?:follow.?up|contact|call|email|meeting)[^\n]*)',
            r'\*\*.*[Ff]ollow.?up.*\*\*[^\n]*\n([^\n#]*(?:- \[ \])[^\n]*)',
        ]
        
        for pattern in followup_patterns:
            matches = re.findall(pattern, summary_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.strip():
                    actions['follow_up_needed'].append(f"[{category_name}] {match.strip()}")
    
    # Remove duplicates while preserving order
    for key in actions:
        actions[key] = list(dict.fromkeys(actions[key]))
    
    return actions

def generate_multi_category_summary_from_classification(transcript: str, classification_result: Dict, 
                                                      use_openai: bool = False, model: str = None) -> Dict[str, Any]:
    """
    Generate summaries from an existing classification result (e.g., from interactive mode)
    """
    template_engine = SummaryTemplateEngine()
    
    # Extract categories from classification result
    above_threshold_categories = classification_result['classification']['above_threshold_categories']
    confidence_scores = classification_result['classification']['all_confidence_scores']
    
    # Generate summaries for categories above threshold
    category_summaries = {}
    for category_name in above_threshold_categories:
        category = ContentCategory(category_name)
        
        # Get category-specific prompt
        prompt = template_engine.format_prompt(category, transcript)
        
        # Generate summary using appropriate AI service
        if use_openai:
            summary = call_openai(prompt, f"You are an expert at analyzing {category.value} content.", model)
        else:
            summary = call_ollama(prompt, model or "llama3.1")
        
        # Apply post-processing
        summary = template_engine.post_process_summary(category, summary)
        
        category_summaries[category.value] = {
            'summary': summary,
            'confidence': confidence_scores[category_name],
            'key_indicators': []  # Would need to be extracted from classification
        }
    
    # Generate unified action dashboard
    unified_actions = generate_unified_action_dashboard(category_summaries, None)
    
    return {
        'classification': classification_result['classification'],
        'category_summaries': category_summaries,
        'unified_actions': unified_actions,
        'metadata': {
            'confidence_threshold': 0.5,  # Default, should be passed in
            'total_categories_analyzed': len(ContentCategory),
            'categories_above_threshold': len(above_threshold_categories),
            'ai_provider': 'openai' if use_openai else 'ollama',
            'model': model or ('gpt-4o-mini' if use_openai else 'llama3.1'),
            'generated_at': datetime.now().isoformat(),
            'interactive_mode': True,
            'feedback_stats': classification_result.get('metadata', {}).get('feedback_stats', {})
        }
    }

def summarize_general(transcript, use_openai=False, model=None):
    """
    Legacy function maintained for backwards compatibility.
    For new implementations, use generate_multi_category_summary instead.
    """
    # Use the new multi-category system but return only a general summary
    result = generate_multi_category_summary(transcript, use_openai, model, confidence_threshold=0.3)
    
    # If we have category summaries, combine them into a general format
    if result['category_summaries']:
        # Get the highest confidence summary
        best_category = result['classification']['primary_category']
        if best_category in result['category_summaries']:
            return result['category_summaries'][best_category]['summary']
    
    # Fallback to simple summary if no categories detected
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


def save_as_json(content, output_path):
    """Save content as JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

def format_multi_category_output(result: Dict[str, Any], output_format: str) -> str:
    """Format multi-category results for different output formats."""
    if output_format == "json":
        return result
    
    # For text/markdown formats, create a comprehensive document
    lines = []
    
    # Header
    lines.append("# Intelligent Multi-Category Transcript Analysis")
    lines.append("")
    lines.append(f"**Generated:** {result['metadata']['generated_at']}")
    lines.append(f"**AI Provider:** {result['metadata']['ai_provider']} ({result['metadata']['model']})")
    lines.append(f"**Confidence Threshold:** {result['metadata']['confidence_threshold']}")
    lines.append("")
    
    # Classification Results
    lines.append("## Content Classification Results")
    lines.append("")
    lines.append(f"**Primary Category:** {result['classification']['primary_category']}")
    lines.append(f"**Categories Above Threshold:** {len(result['classification']['above_threshold_categories'])}")
    lines.append("")
    lines.append("### Confidence Scores")
    
    for category, confidence in sorted(result['classification']['all_confidence_scores'].items(), 
                                     key=lambda x: x[1], reverse=True):
        status = "✓" if confidence >= result['metadata']['confidence_threshold'] else " "
        lines.append(f"- {status} **{category.replace('_', ' ').title()}**: {confidence:.3f}")
    
    lines.append("")
    
    # Unified Action Dashboard
    if result.get('unified_actions'):
        lines.append("## 🎯 Unified Action Dashboard")
        lines.append("")
        
        for action_type, actions in result['unified_actions'].items():
            if actions:
                lines.append(f"### {action_type.replace('_', ' ').title()}")
                for action in actions:
                    lines.append(f"- {action}")
                lines.append("")
    
    # Category-Specific Summaries
    lines.append("## 📄 Category-Specific Summaries")
    lines.append("")
    
    for category, summary_data in result['category_summaries'].items():
        confidence = summary_data['confidence']
        lines.append(f"### {category.replace('_', ' ').title()} (Confidence: {confidence:.3f})")
        lines.append("")
        
        # Add key indicators
        if summary_data.get('key_indicators'):
            lines.append("**Key Indicators:**")
            for indicator in summary_data['key_indicators']:
                lines.append(f"- {indicator}")
            lines.append("")
        
        # Add the summary content
        lines.append(summary_data['summary'])
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Total Categories Analyzed:** {result['metadata']['total_categories_analyzed']}")
    lines.append(f"- **Categories Above Threshold:** {result['metadata']['categories_above_threshold']}")
    
    if result.get('unified_actions'):
        total_actions = sum(len(actions) for actions in result['unified_actions'].values())
        lines.append(f"- **Total Action Items Extracted:** {total_actions}")
    
    return "\n".join(lines)

def save_as_text(content, output_path, style, transcript_file):
    """Save content as plain text file with header."""
    with open(output_path, 'w', encoding='utf-8') as f:
        if isinstance(content, dict):  # Smart summary
            f.write(f"Smart Summary of: {os.path.basename(transcript_file)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            # Convert markdown to plain text (remove markdown formatting)
            plain_content = content.replace('**', '').replace('### ', '').replace('## ', '').replace('# ', '')
            f.write(plain_content)
        else:
            f.write(f"Summary of: {os.path.basename(transcript_file)}\n")
            f.write(f"Style: {style.title()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(content)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced transcript summarization with intelligent multi-category classification",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("transcript_file", help="Path to the transcript file")
    parser.add_argument("-s", "--style", choices=["general", "meeting", "smart"], 
                       default="smart", help="Summary style: smart (intelligent multi-category), general, meeting (default: smart)")
    parser.add_argument("-f", "--format", choices=["txt", "md", "pdf", "json"], 
                       default="md", help="Output format (default: md)")
    parser.add_argument("-o", "--output", help="Output file path (default: append _summary to input filename)")
    parser.add_argument("--provider", choices=["ollama", "openai"], 
                       default="ollama", help="AI provider (default: ollama)")
    parser.add_argument("--model", help="Specific model to use (default: llama3.1 for ollama, gpt-4o-mini for openai)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, 
                       help="Minimum confidence threshold for category inclusion (default: 0.5)")
    parser.add_argument("--show-classification", action="store_true", 
                       help="Display classification results and confidence scores")
    parser.add_argument("--categories-only", action="store_true", 
                       help="Only show classification results without generating summaries")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable interactive mode for uncertain classifications")
    parser.add_argument("--context-hints", 
                       help="Provide context hints for ambiguous terms (format: term1=meaning1,term2=meaning2)")
    
    args = parser.parse_args()
    
    # Check provider availability
    if args.provider == "ollama" and not ollama_available:
        print("Error: Ollama is not running. Please start Ollama or use --provider openai")
        print("To start Ollama: ollama serve")
        sys.exit(1)
    
    if args.provider == "openai" and not openai_client:
        azure_creds = get_azure_openai_credentials()
        if azure_creds:
            print("Error: Azure OpenAI credentials found but client initialization failed. Check your Azure configuration.")
        else:
            print("Error: No OpenAI credentials found. Please set either:")
            print("  - OPENAI_API_KEY for regular OpenAI, or")
            print("  - Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc.) in .env file")
            print("  - Or use --provider ollama")
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
    
    if args.style == "smart":
        print(f"Running intelligent multi-category analysis using {provider_name} ({model_name})...")
        print(f"Confidence threshold: {args.confidence_threshold}")
        
        try:
            if args.interactive:
                # Use interactive classification
                interactive_classifier = InteractiveClassifier(args.confidence_threshold)
                classification_result = interactive_classifier.classify_with_user_input(transcript, interactive=True)
                
                # Convert back to the expected format for summary generation
                result = generate_multi_category_summary_from_classification(
                    transcript, classification_result, use_openai, args.model
                )
            else:
                # Apply context hints if provided
                classifier = ContentClassifier(args.confidence_threshold)
                if args.context_hints:
                    add_context_hints(classifier, args.context_hints)
                    print(f"📝 Applied context hints: {args.context_hints}")
                
                result = generate_multi_category_summary(
                    transcript, use_openai, args.model, args.confidence_threshold, classifier
                )
            
            # Show classification results if requested
            if args.show_classification or args.categories_only:
                print("\n" + "="*60)
                print("CONTENT CLASSIFICATION RESULTS")
                print("="*60)
                print(f"Primary Category: {result['classification']['primary_category']}")
                print(f"Categories Above Threshold ({args.confidence_threshold}): {len(result['classification']['above_threshold_categories'])}")
                
                print("\nAll Category Confidence Scores:")
                for category, confidence in sorted(result['classification']['all_confidence_scores'].items(), 
                                                key=lambda x: x[1], reverse=True):
                    status = "✓" if confidence >= args.confidence_threshold else " "
                    print(f"  {status} {category:25} {confidence:.3f}")
                
                print(f"\nSummaries will be generated for {len(result['classification']['above_threshold_categories'])} categories")
            
            if args.categories_only:
                return
                
            summary_content = format_multi_category_output(result, args.format)
            
        except Exception as e:
            print(f"Error generating multi-category summary: {str(e)}")
            sys.exit(1)
    else:
        print(f"Generating {args.style} summary using {provider_name} ({model_name})...")
        
        try:
            if args.style == "meeting":
                summary_content = summarize_meeting(transcript, use_openai, args.model)
            else:
                summary_content = summarize_general(transcript, use_openai, args.model)
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.transcript_file)[0]
        if args.style == "smart":
            output_path = f"{base_name}_smart_summary.{args.format}"
        else:
            output_path = f"{base_name}_summary.{args.format}"
    
    # Save summary based on format
    try:
        if args.format == "json":
            save_as_json(summary_content, output_path)
        elif args.format == "pdf":
            if args.style == "smart" and isinstance(summary_content, dict):
                # Convert smart summary to text for PDF
                text_content = format_multi_category_output(summary_content, "txt")
                save_as_pdf(text_content, output_path)
            else:
                save_as_pdf(summary_content, output_path)
        elif args.format == "md":
            if args.style == "smart" and isinstance(summary_content, dict):
                # Already formatted as markdown
                save_as_markdown(summary_content, output_path)
            else:
                save_as_markdown(summary_content, output_path)
        else:  # txt
            if args.style == "smart" and isinstance(summary_content, dict):
                text_content = format_multi_category_output(summary_content, "txt")
                save_as_text(text_content, output_path, args.style, args.transcript_file)
            else:
                save_as_text(summary_content, output_path, args.style, args.transcript_file)
        
        print(f"\n✅ Summary saved to: {output_path}")
        
        if args.style == "smart" and isinstance(summary_content, dict):
            num_categories = len(summary_content.get('category_summaries', {}))
            print(f"📊 Generated {num_categories} category-specific summaries")
            if summary_content.get('unified_actions', {}):
                total_actions = sum(len(actions) for actions in summary_content['unified_actions'].values())
                print(f"🎯 Extracted {total_actions} action items across all categories")
        
    except Exception as e:
        print(f"Error writing summary file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()