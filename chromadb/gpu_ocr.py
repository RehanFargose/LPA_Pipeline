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

# Initialize EasyOCR once
USE_GPU = torch.cuda.is_available()
print(f"🚀 GPU Available: {USE_GPU}")
reader = easyocr.Reader(['mr', 'en'], gpu=USE_GPU)

def calculate_page_score(results):
    """Returns a quality score from 0-100 based on OCR confidence."""
    if not results:
        return 0
    
    confidences = []
    for res in results:
        if len(res) == 3:
            confidences.append(res[2])
        else:
            confidences.append(0.9)
            
    avg_conf = np.mean(confidences) * 100
    return round(avg_conf, 2)

def run_ocr():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)
    
    pdf_path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path: 
        print("No file selected.")
        return

    # --- UPDATED: Folder Logic ---
    # 1. Target the existing OCR_Samples/Text_Conversion folder
    base_output_dir = os.path.join("OCR_Samples", "Text_Conversion")
    
    # 2. Get the filename without .pdf
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 3. Create a subfolder specifically for this file
    output_folder = os.path.join(base_output_dir, pdf_name)
    
    # 4. Ensure the path exists (creates Text_Conversion and the subfolder)
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Saving all outputs to: {output_folder}")

    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        full_text = []

        print(f"\n--- OCR Quality Report for {pdf_name} ---")

        for i, page in enumerate(pages):
            img_np = np.array(page)
            results = reader.readtext(img_np, detail=1, paragraph=True)
            
            score = calculate_page_score(results)
            status = "GOOD" if score > 50 else "POOR/IMAGE"
            print(f"📄 Page {i+1}: Quality Score = {score}/100 [%] -> {status}")

            if score < 15:
                print(f"  ⚠️ Skipping Page {i+1}: Likely a photo or unreadable scan.")
                final_page_text = "[Page Skipped: Low Quality/Image]"
            else:
                page_paragraphs = [res[1] for res in results]
                final_page_text = "\n".join(page_paragraphs)

            full_text.append(f"=== Page {i+1} (Score: {score}) ===\n{final_page_text}")
            
            # Memory Cleanup
            if USE_GPU:
                torch.cuda.empty_cache()
            gc.collect()

        # --- UPDATED: Save Output ---
        output_filename = f"{pdf_name}_Scored_Text.txt"
        final_output_path = os.path.join(output_folder, output_filename)
        
        with open(final_output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
        print(f"\n✨ Process complete. File saved at: {final_output_path}")
        messagebox.showinfo("Success", f"OCR Saved to:\n{final_output_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    run_ocr()