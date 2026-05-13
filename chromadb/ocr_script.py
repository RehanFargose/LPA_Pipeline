import os
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import easyocr
import torch
import gc
import warnings

# OCR_Samples/Diksha Parmar VS Haresh Parmar New OCR.pdf
# Suppress the 'pin_memory' warning since we are sticking with CPU
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

# ==== Configuration ====
# Using forward slashes for better compatibility with MINGW64/Git Bash
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
POPPLER_PATH = r"C:/poppler-24.08.0/Library/bin"

# Explicitly forcing CPU for now
print("⏳ Initializing EasyOCR on CPU...")
reader = easyocr.Reader(['mr', 'en'], gpu=False)

def is_low_quality(text):
    """Heuristic to check if Tesseract baseline failed."""
    text = text.strip()
    if len(text) < 60: 
        return True
    junk_indicators = ['|', '°', '—', '_', '']
    if sum(text.count(j) for j in junk_indicators) > 8:
        return True
    return False

def ocr_single_pdf(pdf_path):
    # Standardize path for Windows/Bash cross-compatibility
    pdf_path = os.path.normpath(pdf_path)
    print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        full_text = []

        for i, page in enumerate(pages):
            # 1. Tesseract Attempt (Fastest)
            tess_text = pytesseract.image_to_string(page, lang='mar+eng', config='--psm 3')

            if is_low_quality(tess_text):
                print(f"  - Page {i+1}: Scan/Image detected. Running filtered EasyOCR...")
                img_np = np.array(page)
                
                # detail=1 gives coordinates and confidence scores
                results = reader.readtext(img_np, detail=1, paragraph=True)
                
                page_paragraphs = []
                for res in results:
                    # FIX: Flexible unpacking to handle both (bbox, text, prob) and (bbox, text)
                    if len(res) == 3:
                        bbox, text, prob = res
                    else:
                        bbox, text = res
                        prob = 0.9  # Default confidence if not provided for paragraph block
                    
                    # Filter: Only keep text with > 20% confidence to skip photos/noise
                    if prob > 0.20:
                        page_paragraphs.append(text)
                
                final_page_text = "\n".join(page_paragraphs)
                
                # Clean up memory
                gc.collect()
            else:
                final_page_text = tess_text

            full_text.append(f"=== Page {i+1} ===\n{final_page_text}")
            print(f"✅ Finished Page {i+1}/{len(pages)}")

        # Save Output
        output_filename = os.path.splitext(pdf_path)[0] + "_clean_cpu_text.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
        print(f"\n✨ Process complete: {output_filename}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Remove quotes and normalize the path for Git Bash
    file_path = input("Enter PDF path: ").strip().replace('"', '').replace("'", "")
    if os.path.exists(file_path):
        ocr_single_pdf(file_path)
    else:
        print(f"Invalid path: {file_path}")
        print(f"Current Working Directory: {os.getcwd()}")