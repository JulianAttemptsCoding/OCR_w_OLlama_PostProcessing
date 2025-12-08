import asyncio
import edge_tts
import os
import textwrap

VOICE = "en-US-GuyNeural"
INPUT_FILE = "final_audio_clean.txt"
OUTPUT_DIR = "audio_parts"
CHUNK_SIZE = 4000  # Characters per chunk (approx 5-10 mins of audio)

async def generate_audio():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks to avoid timeouts and make it manageable
    # We split by paragraphs (double newlines) first to avoid cutting sentences
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < CHUNK_SIZE:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk)

    print(f"Generated {len(chunks)} chunks. Starting conversion...")

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        output_file = os.path.join(OUTPUT_DIR, f"part_{i+1:03d}.mp3")
        print(f"Generating {output_file} ({len(chunk)} chars)...")
        
        communicate = edge_tts.Communicate(chunk, VOICE)
        await communicate.save(output_file)

    print("Audio generation complete!")

if __name__ == "__main__":
    asyncio.run(generate_audio())

