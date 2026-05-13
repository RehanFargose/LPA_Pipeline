import os
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import easyocr
import torch
import gc
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox

# ==== Configuration ====
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
POPPLER_PATH = r"C:/poppler-24.08.0/Library/bin"

# Force GPU Detection
USE_GPU = torch.cuda.is_available()
print(f"🚀 GPU Available: {USE_GPU}")
if USE_GPU:
    print(f"🎸 Using Device: {torch.cuda.get_device_name(0)}")

# Initialize EasyOCR once
print("⏳ Loading OCR models into memory...")
reader = easyocr.Reader(['mr', 'en'], gpu=USE_GPU)

def is_low_quality(text):
    text = text.strip()
    if len(text) < 60: 
        return True
    junk_indicators = ['|', '°', '—', '_']
    if sum(text.count(j) for j in junk_indicators) > 8:
        return True
    return False

def run_ocr():
    # Create a hidden Tkinter root to show the file picker
    root = tk.Tk()
    root.withdraw() 
    
    # Open File Picker
    pdf_path = filedialog.askopenfilename(
        title="Select PDF for OCR",
        filetypes=[("PDF files", "*.pdf")]
    )
    
    if not pdf_path:
        print("No file selected. Exiting.")
        return

    print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        full_text = []

        for i, page in enumerate(pages):
            # 1. Tesseract Attempt
            tess_text = pytesseract.image_to_string(page, lang='mar+eng', config='--psm 3')

            if is_low_quality(tess_text):
                print(f"  - Page {i+1}: Low quality. Switching to EasyOCR (GPU)...")
                img_np = np.array(page)
                
                # GPU inference happens here
                results = reader.readtext(img_np, detail=1, paragraph=True)
                
                page_paragraphs = [res[1] for res in results if (res[2] if len(res)==3 else 0.9) > 0.20]
                final_page_text = "\n".join(page_paragraphs)
                
                # Clear VRAM after heavy lifting
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                final_page_text = tess_text

            full_text.append(f"=== Page {i+1} ===\n{final_page_text}")
            print(f"✅ Finished Page {i+1}/{len(pages)}")

        # Save Output
        output_filename = os.path.splitext(pdf_path)[0] + "_GPU_text.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
        # messagebox.showinfo("Success", f"OCR Complete!\nSaved to: {os.path.basename(output_filename)}")
        print(f"\n✨ Process complete: {output_filename}")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_ocr()