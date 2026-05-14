import fitz  # PyMuPDF
import re
import io
import os  
import time
import gc         
import torch
from docx import Document
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from RAG_Stages.local_translator_utils import local_translate


# Ensures consistent results
DetectorFactory.seed = 0

# Mapping langdetect codes to NLLB codes
LANG_MAP = {
    'mr': 'mar_Deva',
    'hi': 'hin_Deva',
    'gu': 'guj_Gujr'
}

def save_translation(text, base_name):
    """
    Saves the provided text into 'translations' folder in both .md and .txt formats.
    """
    output_dir = "translations"
    os.makedirs(output_dir, exist_ok=True)

    # Clean the base_name to avoid trailing spaces in filenames
    clean_name = base_name.strip()
    txt_path = os.path.join(output_dir, f"{clean_name}_translated.txt")
    md_path = os.path.join(output_dir, f"{clean_name}_translated.md")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Translated Document: {clean_name}\n\n{text}")
    
    print(f"✅ Saved translations to: {output_dir}")

def legal_cleaning(text):
    """Enhanced cleaning for Indian Law Reports."""
    text = re.sub(r'^\s*[A-H]\s*$', '', text, flags=re.MULTILINE)
    headers = [r'\[\d{4}\]\s*\d+\s*S\.C\.R\.', r'SUPREME\s*COURT\s*REPORTS', r'HIGH\s*COURT\s*REPORTS']
    for pattern in headers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = "".join(char for char in text if char.isprintable() or char in ['\n', '\t'])
    text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])
    return text.strip()

def process_document(uploaded_file, translate_to_en=True):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    input_file_name = uploaded_file.name.split('.')[0]
    print(f"Processing file: {input_file_name}.{file_extension}")
    raw_text = ""

    if file_extension == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc: raw_text += page.get_text()
    elif file_extension == "docx":
        doc = Document(io.BytesIO(uploaded_file.read()))
        raw_text = "\n".join([para.text for para in doc.paragraphs])
    elif file_extension == "txt":
        raw_text = uploaded_file.read().decode("utf-8")
    else:
        return None

    cleaned_text = legal_cleaning(raw_text)
    
    if translate_to_en:
        try:
            detected_lang = detect(cleaned_text[:1000])
            if detected_lang in LANG_MAP:
                nllb_code = LANG_MAP[detected_lang]
                print(f"🚀 Detected {detected_lang}. Using local GPU translation ({nllb_code})...")
                
                # Clear VRAM before translation if OCR was running
                gc.collect()
                torch.cuda.empty_cache()

                # --- USE LOCAL TRANSLATOR ---
                cleaned_text = local_translate(cleaned_text, src_lang=nllb_code)
                
                save_translation(cleaned_text, input_file_name)
            elif detected_lang == 'en':
                print("Document is already in English.")
            else:
                print(f"Detected {detected_lang}, but no local model mapping found.")
        except Exception as e:
            print(f"Local translation failed: {e}")
    
    
    return cleaned_text, input_file_name