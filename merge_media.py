#!/usr/bin/env python3
"""
Enhanced Audio/Video Merging Tool

Merges multiple audio/video files using ffmpeg and organizes segments into folders.
Supports auto-detection of segment files and intelligent naming.
"""

import os
import sys
import re
import argparse
import subprocess
import shutil
from pathlib import Path

def find_ffmpeg():
    """Find ffmpeg executable"""
    # Check if ffmpeg is in PATH
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    
    # Check local ffmpeg (from previous download)
    local_ffmpeg = Path('./ffmpeg')
    if local_ffmpeg.exists():
        return str(local_ffmpeg.resolve())
    
    print("Error: ffmpeg not found. Please install ffmpeg:")
    print("  sudo apt install ffmpeg")
    print("  # or download from https://ffmpeg.org/")
    sys.exit(1)

def extract_base_name(filename):
    """Extract base name by removing segment indicators"""
    # Remove file extension
    name = Path(filename).stem
    
    # Special case handling first - if this is a pure base name without indicators, return as-is
    if not any(indicator in name for indicator in ['(', ')', '_part', '_segment', 'merged']):
        return name
    
    # Specific segment patterns to remove
    if ' (1-2)' in name or ' (2-2)' in name:
        name = re.sub(r'\s*\(\d+-\d+\)$', '', name)
    elif '(1 of' in name or '(2 of' in name:
        name = re.sub(r'\s*\(\d+\s*of\s*\d+\)$', '', name)
    elif '_part' in name.lower():
        name = re.sub(r'\s*_part\d+$', '', name, flags=re.IGNORECASE)
    elif '_segment' in name.lower():
        name = re.sub(r'\s*_segment\d+$', '', name, flags=re.IGNORECASE)
    elif '(merged)' in name.lower():
        name = re.sub(r'\s*\(merged\)$', '', name, flags=re.IGNORECASE)
    elif name.endswith(')') and '(' in name:
        # Check if it's a simple number in parentheses at the end
        if re.search(r'\s*\(\d+\)$', name):
            name = re.sub(r'\s*\(\d+\)$', '', name)
    
    return name.strip()

def find_segment_files(base_name, directory="."):
    """Find all segment files that match the base name"""
    directory = Path(directory)
    supported_extensions = ['.m4a', '.wav', '.mp3', '.mp4', '.mov']
    
    segment_files = []
    
    # Look for files that start with the base name
    for ext in supported_extensions:
        pattern = f"{base_name}*{ext}"
        files = list(directory.glob(pattern))
        segment_files.extend(files)
    
    # Filter out files that are exact matches or already merged
    filtered_files = []
    for f in segment_files:
        name_stem = f.stem
        # Keep files that have segment indicators but exclude exact matches
        if (name_stem != base_name and 
            not name_stem.endswith(' (merged)') and
            extract_base_name(f.name) == base_name):
            filtered_files.append(f)
    
    segment_files = filtered_files
    
    # Sort files naturally (handle numbers correctly)
    def natural_sort_key(path):
        # Extract numbers from filename for proper sorting
        parts = re.split(r'(\d+)', str(path))
        return [int(part) if part.isdigit() else part for part in parts]
    
    return sorted(segment_files, key=natural_sort_key)

def merge_files(input_files, output_file):
    """Merge multiple audio/video files using ffmpeg"""
    if not input_files:
        print("Error: No input files provided")
        return False
    
    if len(input_files) == 1:
        print("Only one file provided, copying instead of merging...")
        shutil.copy2(input_files[0], output_file)
        return True
    
    ffmpeg = find_ffmpeg()
    
    # Create temporary file list for ffmpeg concat
    temp_list = Path("temp_file_list.txt")
    
    try:
        # Write file list for ffmpeg concat demuxer
        with open(temp_list, 'w') as f:
            for file_path in input_files:
                # Use absolute paths and escape single quotes
                abs_path = Path(file_path).resolve()
                f.write(f"file '{abs_path}'\n")
        
        # Run ffmpeg concat
        cmd = [
            ffmpeg,
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_list),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            str(output_file),
            '-y'  # Overwrite output file
        ]
        
        print(f"Merging {len(input_files)} files...")
        print(f"Output: {output_file}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Merge completed successfully!")
            return True
        else:
            print(f"❌ ffmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return False
    finally:
        # Clean up temp file
        if temp_list.exists():
            temp_list.unlink()

def organize_segments(segment_files, base_name):
    """Move segment files to a dedicated folder"""
    if not segment_files:
        return None
    
    # Detect if we're working with files in an inputs/date/ structure
    segments_dir = None
    for file_path in segment_files:
        file_path = Path(file_path)
        # Check if the file is in an inputs/date/ structure
        if len(file_path.parts) >= 2 and file_path.parts[-3] == "inputs":
            # Extract the date folder (e.g., "2025-07-15")
            date_folder = file_path.parts[-2]
            inputs_date_dir = file_path.parent  # This is the inputs/date/ directory
            segments_dir = inputs_date_dir / f"{base_name}_segments"
            break
    
    # If no inputs/date/ structure found, use current directory
    if segments_dir is None:
        segments_dir = Path(f"{base_name}_segments")
    
    segments_dir.mkdir(exist_ok=True)
    
    moved_files = []
    for file_path in segment_files:
        file_path = Path(file_path)
        destination = segments_dir / file_path.name
        
        try:
            shutil.move(str(file_path), str(destination))
            moved_files.append(destination)
            print(f"  📁 Moved {file_path.name} → {segments_dir.name}/")
        except Exception as e:
            print(f"⚠️  Warning: Could not move {file_path.name}: {e}")
    
    return segments_dir if moved_files else None

def auto_detect_and_merge(base_pattern, directory="."):
    """Auto-detect segment files and merge them"""
    directory = Path(directory)
    
    # Check if base_pattern has obvious segment indicators
    has_segment_indicators = any(pattern in base_pattern.lower() for pattern in [
        '(', ')', '_part', '_segment', 'merged'
    ])
    
    # If base_pattern looks like a full filename or has segment indicators, extract the base
    if '.' in base_pattern or has_segment_indicators:
        base_name = extract_base_name(base_pattern)
    else:
        # Use the pattern as-is for searching
        base_name = base_pattern
    
    print(f"🔍 Looking for segments matching: '{base_name}'")
    
    # Find all segment files
    segment_files = find_segment_files(base_name, directory)
    
    if not segment_files:
        print(f"❌ No segment files found matching '{base_name}'")
        print("   Try specifying files manually: python merge_media.py file1.m4a file2.m4a")
        return False
    
    print(f"📋 Found {len(segment_files)} segment files:")
    for i, file_path in enumerate(segment_files, 1):
        print(f"  {i}. {file_path.name}")
    
    # Determine output extension from first file
    first_file = segment_files[0]
    output_extension = first_file.suffix
    output_file = directory / f"{base_name}{output_extension}"
    
    # Merge files
    if merge_files(segment_files, output_file):
        # Organize segments into folder
        segments_dir = organize_segments(segment_files, base_name)
        
        print(f"\n🎉 Successfully merged into: {output_file.name}")
        if segments_dir:
            print(f"📁 Original segments moved to: {segments_dir.name}/")
        return True
    
    return False

def manual_merge(file_list, output_file=None):
    """Manually specify files to merge"""
    # Validate all files exist
    input_files = []
    for file_path in file_list:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        input_files.append(path)
    
    # Auto-generate output filename if not specified
    if not output_file:
        first_file = input_files[0]
        base_name = extract_base_name(first_file.name)
        output_file = first_file.parent / f"{base_name}{first_file.suffix}"
    else:
        output_file = Path(output_file)
    
    print(f"📋 Merging {len(input_files)} files:")
    for i, file_path in enumerate(input_files, 1):
        print(f"  {i}. {file_path.name}")
    
    # Merge files
    if merge_files(input_files, output_file):
        # Extract base name for organizing segments
        base_name = extract_base_name(input_files[0].name)
        
        # Organize segments into folder
        segments_dir = organize_segments(input_files, base_name)
        
        print(f"\n🎉 Successfully merged into: {output_file.name}")
        if segments_dir:
            print(f"📁 Original segments moved to: {segments_dir.name}/")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Audio/Video Merging Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect segments by base name
  python merge_media.py "Hunt Working Session 7.15"
  
  # Manual file specification
  python merge_media.py file1.m4a file2.m4a file3.m4a
  
  # Specify output file
  python merge_media.py file1.m4a file2.m4a -o merged.m4a
  
  # Process files in different directory
  python merge_media.py "Meeting" -d inputs/2025-07-15/

Supported formats: .m4a, .wav, .mp3, .mp4, .mov
        """
    )
    
    parser.add_argument('files', nargs='+', 
                       help='Base name for auto-detection or list of files to merge')
    parser.add_argument('-o', '--output', 
                       help='Output file name (auto-generated if not specified)')
    parser.add_argument('-d', '--directory', default='.', 
                       help='Directory to search for files (default: current)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # If only one argument and it doesn't exist as a file, assume it's a base name for auto-detection
    if len(args.files) == 1 and not Path(args.files[0]).exists():
        # Auto-detection mode
        base_pattern = args.files[0]
        if args.dry_run:
            print("🧪 DRY RUN - No files will be modified")
            # Use the same logic as auto_detect_and_merge for consistency
            has_segment_indicators = any(pattern in base_pattern.lower() for pattern in [
                '(', ')', '_part', '_segment', 'merged'
            ])
            
            if '.' in base_pattern or has_segment_indicators:
                search_name = extract_base_name(base_pattern)
            else:
                search_name = base_pattern
            
            segment_files = find_segment_files(search_name, args.directory)
            if segment_files:
                print(f"Would merge {len(segment_files)} files:")
                for f in segment_files:
                    print(f"  - {f.name}")
            else:
                print(f"No segments found for '{search_name}'")
        else:
            success = auto_detect_and_merge(base_pattern, args.directory)
            sys.exit(0 if success else 1)
    else:
        # Manual file specification
        if args.dry_run:
            print("🧪 DRY RUN - No files will be modified")
            print(f"Would merge {len(args.files)} files:")
            for f in args.files:
                print(f"  - {f}")
        else:
            success = manual_merge(args.files, args.output)
            sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()