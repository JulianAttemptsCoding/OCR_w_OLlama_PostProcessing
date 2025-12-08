import re
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def clean_audio_text(input_file: str, output_file: str):
    """
    Cleans the audio text file by removing artifacts, forms, chatter, and repetition.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    previous_paragraph = ""
    
    # Regex patterns
    underscore_pattern = re.compile(r'_{3,}')  # Matches 3 or more underscores (forms)
    chatter_pattern = re.compile(r'^(The output is|Here is|I have corrected|The text has been|This output).*', re.IGNORECASE)
    page_num_only = re.compile(r'^\s*\d+\s*$')
    
    # End of  markers (Forms that appear at the end)
    end_of_pattern = re.compile(r'Phoenix Consulting Group Contact Data File', re.IGNORECASE)
    
    # Specific typo fixes
    typo_map = {
        "Thar's": "That's",
        "chat was": "that was", 
    }

    logging.info("Processing lines...")
    
    buffer_paragraph = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # 1. Check for End of 
        if end_of_pattern.search(line):
            logging.info("Found end-of- marker. Stopping processing.")
            break

        # 2. Skip empty lines (we'll handle paragraph spacing later)
        if not line:
            if buffer_paragraph:
                # Process the buffered paragraph
                processed_para = process_paragraph(" ".join(buffer_paragraph), underscore_pattern, chatter_pattern, typo_map)
                if processed_para and processed_para != previous_paragraph:
                    cleaned_lines.append(processed_para)
                    cleaned_lines.append("") # Add blank line after paragraph
                    previous_paragraph = processed_para
                buffer_paragraph = []
            continue

        # 3. Filter out Page Numbers immediately
        if page_num_only.match(line):
            continue
            
        # 4. Filter out Forms/Underscores immediately
        if underscore_pattern.search(line):
            continue
            
        # 5. Filter out Chatter immediately
        if chatter_pattern.match(line):
            continue

        # 6. Clean Artifacts from the line
        line = line.replace('|', '')
        line = line.replace('~', '')
        line = line.replace('°', '')
        line = line.replace('¢', '')
        line = line.replace('©', '') 
        
        # 7. Accumulate into paragraph buffer
        if line.endswith('-'):
            buffer_paragraph.append(line[:-1])
        else:
            buffer_paragraph.append(line)

    # Process any remaining buffer
    if buffer_paragraph:
        processed_para = process_paragraph(" ".join(buffer_paragraph), underscore_pattern, chatter_pattern, typo_map)
        if processed_para and processed_para != previous_paragraph:
            cleaned_lines.append(processed_para)

    logging.info(f"Writing cleaned text to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + "\n")
            
    logging.info("Cleanup complete.")

def process_paragraph(text, underscore_pattern, chatter_pattern, typo_map):
    """
    Helper to clean a full paragraph string.
    """
    # Apply typo fixes
    for bad, good in typo_map.items():
        text = text.replace(bad, good)
        
    # Double check for chatter that might be embedded
    if chatter_pattern.match(text):
        return None
        
    # Double check for forms
    if underscore_pattern.search(text):
        return None
        
    return text.strip()

if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Cleanup Audio Text")
    parser.add_argument("--input", default="final_audio.txt", help="Input text file")
    parser.add_argument("--output", default="final_audio_clean.txt", help="Output text file")
    args = parser.parse_args()
    
    clean_audio_text(args.input, args.output)

