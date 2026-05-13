import os
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import easyocr
import torch
import gc
import tkinter as tk
from tkinter import filedialog, messagebox

# ==== Configuration ====
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
POPPLER_PATH = r"C:/poppler-24.08.0/Library/bin"

USE_GPU = torch.cuda.is_available()
reader = easyocr.Reader(['mr', 'en'], gpu=USE_GPU)

def calculate_page_score(results):
    """Returns a quality score from 0-100 based on OCR confidence."""
    if not results:
        return 0
    
    # Average the confidence scores provided by EasyOCR
    confidences = []
    for res in results:
        if len(res) == 3:
            confidences.append(res[2])
        else:
            confidences.append(0.9) # Default for paragraph mode if missing
            
    avg_conf = np.mean(confidences) * 100
    return round(avg_conf, 2)

def run_ocr():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)
    
    pdf_path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path: return

    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        full_text = []

        print(f"\n--- OCR Quality Report for {os.path.basename(pdf_path)} ---")

        for i, page in enumerate(pages):
            img_np = np.array(page)
            
            # We run EasyOCR to get the text and the metadata
            results = reader.readtext(img_np, detail=1, paragraph=True)
            
            # Calculate Score
            score = calculate_page_score(results)
            
            # Log the score to terminal so you can see the results
            status = "GOOD" if score > 50 else "POOR/IMAGE"
            print(f"📄 Page {i+1}: Quality Score = {score}/100 [%] -> {status}")

            # Decide whether to keep or skip based on a threshold (e.g., 20)
            if score < 15:
                print(f"  ⚠️ Skipping Page {i+1}: Likely a photo or unreadable scan.")
                final_page_text = "[Page Skipped: Low Quality/Image]"
            else:
                page_paragraphs = [res[1] for res in results]
                final_page_text = "\n".join(page_paragraphs)

            full_text.append(f"=== Page {i+1} (Score: {score}) ===\n{final_page_text}")
            
            if USE_GPU:
                torch.cuda.empty_cache()
            gc.collect()

        # Save Output
        output_filename = os.path.splitext(pdf_path)[0] + "_Scored_Text.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
        print(f"\n✨ Process complete. Check the console for page scores!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_ocr()