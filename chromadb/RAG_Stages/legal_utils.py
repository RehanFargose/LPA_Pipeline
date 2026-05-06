import fitz  # PyMuPDF
import re
import io
from docx import Document
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Ensures consistent results
DetectorFactory.seed = 0

def legal_cleaning(text):
    """
    Enhanced cleaning for Indian Law Reports:
    Removes margin ladders (A-H), page headers, and numbers.
    """
    # 1. Remove Margin Ladders (A-H) used for line referencing
    text = re.sub(r'^\s*[A-H]\s*$', '', text, flags=re.MULTILINE)

    # 2. Remove Common SCR/ILR Headers
    headers = [
        r'\[\d{4}\]\s*\d+\s*S\.C\.R\.',
        r'SUPREME\s*COURT\s*REPORTS',
        r'HIGH\s*COURT\s*REPORTS',
        r'CASE\s*LAW\s*REFERENCE',
        r'DIGEST\s*OF\s*CASES'
    ]
    for pattern in headers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Remove Standalone Page Numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 4. Whitespace and Character Normalization
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = "".join(char for char in text if char.isprintable() or char in ['\n', '\t'])

    # 5. Final structure preservation
    text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])

    return text.strip()

def process_document(uploaded_file, translate_to_en=True):
    """
    Detects file type, extracts text, and applies legal cleaning.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    input_file_name = uploaded_file.name.split('.')[0]
    print(f"Processing file: {input_file_name}.{file_extension}")
    raw_text = ""

    if file_extension == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
    
    elif file_extension == "docx":
        doc = Document(io.BytesIO(uploaded_file.read()))
        raw_text = "\n".join([para.text for para in doc.paragraphs])
    
    else:
        return None

    # Clean text 1st
    cleaned_text = legal_cleaning(raw_text)
    
    # Translate if it's not in English
    if translate_to_en:
        try:
            detected_lang = detect(cleaned_text[:1000])
            
            # If not in english, translate to English
            if detected_lang != 'en':
                print(f"Detected {detected_lang}, initiating translation...")
                # GoogleTranslator handles 'auto' detection for Hindi, Marathi, Gujarati, etc.
                #  ~5000 character limit per request, so chunk if necessary
                translator = GoogleTranslator(source='auto', target='en')
                
                # Chunking logic for long legal documents
                chunks = [cleaned_text[i:i+4500] for i in range(0, len(cleaned_text), 4500)]
                translated_chunks = [translator.translate(ch) for ch in chunks]
                cleaned_text = " ".join(translated_chunks)
        except Exception as e:
            print(f"Translation failed: {e}")
            
    
    return cleaned_text, input_file_name