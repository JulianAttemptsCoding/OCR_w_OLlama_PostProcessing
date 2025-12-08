import argparse
import os
import sys
import logging
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import re
import requests
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Union
import importlib

os.environ['OMP_THREAD_LIMIT'] = '1'

=poppler_path = r"{Input_Path}"
if poppler_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + poppler_path

# Add Tesseract to PATH if not already present
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + tesseract_path

# --- Configuration ---

@dataclass
class ProcessingConfig:
    input_file: str
    output_file: str
    llm_framework: str
    model_name: str
    preprocess: bool
    ocr_config: str
    dpi: int
    top_crop_percentage: int
    max_workers: int
    start_page: int
    end_page: int

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Default configuration
DEFAULT_LLM_FRAMEWORK = "ollama"
DEFAULT_MODEL_NAME = "mistral"
DEFAULT_OUTPUT_FILE = "final_audio.txt"
DEFAULT_OCR_CONFIG = "--oem 3 --psm 6"
DEFAULT_PREPROCESS = True
DEFAULT_TOP_CROP_PERCENTAGE = 7
DEFAULT_INPUT_FILE = "Conf_JN.pdf"
DEFAULT_MAX_WORKERS = 8 

# --- Helper Functions ---

def crop_top_percentage(image: np.ndarray, crop_percentage: int) -> np.ndarray:
    """Crops the top percentage of the image."""
    if crop_percentage <= 0 or crop_percentage >= 100:
        return image

    height = image.shape[0]
    crop_height = int(height * (crop_percentage / 100))
    return image[crop_height:, :]

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Applies basic image preprocessing for better OCR results."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply threshold to create a binary image (binarization)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def perform_ocr(image: Union[np.ndarray, Image.Image], config: str, preprocess: bool, crop_percentage: int) -> str:
    """Performs OCR on an image (numpy array or PIL) and returns the raw text."""
    
    # Convert PIL to Numpy if needed for preprocessing
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    if preprocess:
        processed_img = preprocess_image(image_np)
    else:
        processed_img = image_np
        if len(processed_img.shape) == 3:
             processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    # Crop
    if crop_percentage > 0:
        processed_img = crop_top_percentage(processed_img, crop_percentage)

    # Convert back to PIL for Tesseract (it handles PIL images efficiently)
    pil_img = Image.fromarray(processed_img)

    raw_text = pytesseract.image_to_string(pil_img, config=config)
    return raw_text.strip()

def load_llm_model(model_name: str, framework: str) -> Tuple[Any, Any]:
    """Loads the specified LLM model based on the chosen framework."""
    logger.info(f"Loading LLM model '{model_name}' using framework '{framework}'")
    model = None
    tokenizer = None

    if framework == "transformers":
        try:
            transformers = importlib.import_module("transformers")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        except ImportError:
            logger.error("Transformers library not installed.")
            raise
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise e
    elif framework == "llama_cpp":
        try:
            llama_cpp = importlib.import_module("llama_cpp")
            model = llama_cpp.Llama(model_path=model_name, n_ctx=2048, verbose=False)
        except ImportError:
            logger.error("llama-cpp-python library not installed.")
            raise
        except Exception as e:
            logger.error(f"Error loading llama.cpp model: {e}")
            raise e
    elif framework == "ollama":
        model = model_name
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return model, tokenizer

def create_correction_prompt(raw_text: str) -> str:
    """Creates a strict Few-Shot prompt to force the LLM to output ONLY corrected text."""
    prompt = f"""You are a helpful assistant that corrects OCR text.
Your goal is to fix spelling errors, punctuation mistakes, and character substitutions.
IMPORTANT RULES:
1. Output ONLY the corrected text.
2. Do NOT add any conversational filler like "Here is the corrected text" or "I have fixed the errors".
3. Do NOT add "OCR Text:" or "Corrected Text:" labels.
4. Maintain the original meaning and structure.
5. If the text is empty or just noise, output nothing.

Example 1:
Input: Th quick brwn fx jmps ovr th lzy dg.
Output: The quick brown fox jumps over the lazy dog.

Example 2:
Input: Chaptr 1: The Begining
Output: Chapter 1: The Beginning

Input:
{raw_text}

Output:"""
    return prompt

def get_llm_response(prompt: str, model: Any, tokenizer: Any, framework: str, model_name: str) -> str:
    """Sends the prompt to the loaded LLM and retrieves the response."""
    response_text = ""

    if framework == "transformers":
        # Simplified for brevity
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif framework == "llama_cpp":
        output = model(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["\n\n", "Input:", "Output:"],
        )
        response_text = output['choices'][0]['text']
    elif framework == "ollama":
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
                "stop": ["Input:", "Output:"] 
            }
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise e
    
    return response_text.strip()

def post_process_llm_output(corrected_text: str) -> str:
    """Applies Regex filters to remove artifacts and hallucinations."""
    
    # 1. Remove "Here is the corrected text" and similar chatter
    corrected_text = re.sub(r'^(Here is|This is|I have corrected).*?:', '', corrected_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Remove "OCR Text:" or "Corrected Text:" labels if they slipped through
    corrected_text = re.sub(r'^(OCR Text:|Corrected Text:)', '', corrected_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 3. Remove known hallucinations (e.g., "The quick brown fox...")
    corrected_text = re.sub(r'The quick brown fox jumps over the lazy dog\.?', '', corrected_text, flags=re.IGNORECASE)
    
    # 4. Remove lines that are just page numbers (digits only)
    corrected_text = re.sub(r'^\s*\d+\s*$', '', corrected_text, flags=re.MULTILINE)
    
    # 5. Remove "--- PAGE BREAK ---" if it exists (though we aren't adding it anymore)
    corrected_text = corrected_text.replace("--- PAGE BREAK ---", "")

    # 6. Remove ISBN/Copyright lines
    corrected_text = re.sub(r'^ISBN:.*$', '', corrected_text, flags=re.MULTILINE)
    corrected_text = re.sub(r'^Copyright.*$', '', corrected_text, flags=re.MULTILINE | re.IGNORECASE)

    return corrected_text.strip()

def process_page(page_index: int, page_image: Image.Image, config: ProcessingConfig, model: Any, tokenizer: Any) -> Tuple[int, str]:
    """Processes a single page: OCR -> LLM Correction."""
    logger.info(f"Processing page {page_index}...")
    
    # OCR
    raw_text = perform_ocr(page_image, config=config.ocr_config, preprocess=config.preprocess, crop_percentage=config.top_crop_percentage)
    
    # Skip empty or very short pages (likely blank or just an image)
    if len(raw_text.strip()) < 10:
        logger.info(f"Page {page_index} is empty or too short. Skipping LLM.")
        return page_index, ""

    # LLM
    prompt = create_correction_prompt(raw_text)
    try:
        llm_corrected_text = get_llm_response(prompt, model, tokenizer, config.llm_framework, config.model_name)
    except Exception as e:
        logger.error(f"LLM processing failed for page {page_index}: {e}")
        llm_corrected_text = raw_text # Fallback

    final_text = post_process_llm_output(llm_corrected_text)
    return page_index, final_text

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="OCR PDF and post-process with local LLM for Audio creation.")
    parser.add_argument("input_file", nargs='?', default=DEFAULT_INPUT_FILE, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="Output text file path.")
    parser.add_argument("--llm_framework", choices=["transformers", "llama_cpp", "ollama"], default=DEFAULT_LLM_FRAMEWORK, help="LLM framework to use.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Name or path of the LLM model.")
    parser.add_argument("--no_preprocess", action='store_true', help="Skip image preprocessing.")
    parser.add_argument("--ocr_config", default=DEFAULT_OCR_CONFIG, help="Tesseract OCR configuration string.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF to image conversion.")
    parser.add_argument("--top_crop", type=int, default=DEFAULT_TOP_CROP_PERCENTAGE, 
                        help=f"Percentage of top to crop from each page (default: {DEFAULT_TOP_CROP_PERCENTAGE}%)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of parallel workers.")
    parser.add_argument("--start_page", type=int, default=1, help="Page number to start processing (1-based).")
    parser.add_argument("--end_page", type=int, default=None, help="Page number to stop processing (inclusive).")

    args = parser.parse_args()

    config = ProcessingConfig(
        input_file=args.input_file,
        output_file=args.output,
        llm_framework=args.llm_framework,
        model_name=args.model_name,
        preprocess=not args.no_preprocess,
        ocr_config=args.ocr_config,
        dpi=args.dpi,
        top_crop_percentage=args.top_crop,
        max_workers=args.workers,
        start_page=args.start_page,
        end_page=args.end_page
    )

    logger.info(f"Starting Audio OCR pipeline for: {config.input_file}")
    
    if not os.path.exists(config.input_file):
        logger.error(f"Input file does not exist: {config.input_file}")
        sys.exit(1)

    # 1. Get Page Count
    try:
        info = pdfinfo_from_path(config.input_file)
        total_pdf_pages = info["Pages"]
        logger.info(f"PDF has {total_pdf_pages} total pages.")
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        sys.exit(1)

    # Determine processing range
    start_page = max(1, config.start_page)
    end_page = min(total_pdf_pages, config.end_page) if config.end_page else total_pdf_pages
    
    if start_page > end_page:
        logger.error(f"Start page {start_page} is greater than end page {end_page}.")
        sys.exit(1)
        
    logger.info(f"Processing range: Page {start_page} to {end_page}")

    # Load LLM Model
    try:
        model, tokenizer = load_llm_model(config.model_name, config.llm_framework)
    except Exception as e:
        sys.exit(1)

    # 2. Process in Batches
    BATCH_SIZE = 10
    all_results = []

    for batch_start in range(start_page, end_page + 1, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE - 1, end_page)
        logger.info(f"--- Processing Batch: Pages {batch_start} to {batch_end} ---")

        try:
            # Convert only the current batch of pages
            # Note: convert_from_path uses 1-based indexing for first_page/last_page
            batch_pages = convert_from_path(
                config.input_file, 
                dpi=config.dpi, 
                first_page=batch_start, 
                last_page=batch_end,
                grayscale=True
            )
        except Exception as e:
            logger.error(f"Error converting batch {batch_start}-{batch_end}: {e}")
            continue

        # Process the batch in parallel
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Map futures to page numbers
            future_to_page = {
                executor.submit(process_page, batch_start + i, page, config, model, tokenizer): (batch_start + i)
                for i, page in enumerate(batch_pages)
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    page_num, text = future.result()
                    batch_results.append((page_num, text))
                except Exception as e:
                    logger.error(f"Page processing generated an exception: {e}")
        
        all_results.extend(batch_results)
        
        # Explicitly clear memory
        del batch_pages
        del batch_results

    # Sort results by page index
    all_results.sort(key=lambda x: x[0])
    
    # 3. Smart Joining for Audio Flow
    logger.info("Joining text with smart flow logic...")
    final_text_blocks = []
    
    for i, (page_num, text) in enumerate(all_results):
        if not text:
            continue
            
        # Check if the previous block ended with a sentence terminator
        if final_text_blocks:
            prev_text = final_text_blocks[-1]
            if prev_text:
                if prev_text.endswith('-'):
                     # Hyphenated word split across pages (e.g., "infor-" + "mation")
                     # Remove hyphen and join without space
                     final_text_blocks[-1] = prev_text[:-1] # Remove hyphen from prev
                     join_char = ""
                elif prev_text[-1] not in ['.', '!', '?', '"', "'", ')', ']', '}']:
                    # Previous page ended mid-sentence (likely), join with space
                    join_char = " "
                else:
                    # Previous page ended with punctuation, start new paragraph
                    join_char = "\n\n"
            else:
                 join_char = "\n\n"
        else:
            join_char = ""
            
        final_text_blocks.append(join_char + text)

    full_corrected_text = "".join(final_text_blocks)

    # Save to Output File
    with open(config.output_file, 'w', encoding='utf-8') as f:
        f.write(full_corrected_text)
    logger.info(f"Audio-ready text saved to: {config.output_file}")

    logger.info("OCR and LLM post-processing completed successfully.")

if _name_ == "_main_":

    main()



