import os
import glob

OUTPUT_DIR = "audiobook_parts"
FINAL_OUTPUT = "final_audiobook_complete.mp3"

def merge_audio():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: {OUTPUT_DIR} not found.")
        return

    # Get all mp3 files sorted by name
    mp3_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "part_*.mp3")))
    
    if not mp3_files:
        print("No MP3 files found to merge.")
        return

    print(f"Found {len(mp3_files)} parts. Merging into {FINAL_OUTPUT}...")

    with open(FINAL_OUTPUT, 'wb') as outfile:
        for mp3_file in mp3_files:
            print(f"Appending {mp3_file}...")
            with open(mp3_file, 'rb') as infile:
                outfile.write(infile.read())

    print(f"Successfully created {FINAL_OUTPUT}")

if __name__ == "__main__":
    merge_audio()
