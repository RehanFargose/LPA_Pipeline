import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

def local_translate(text, src_lang="mar_Deva"):
    """
    Dynamically loads NLLB in Half-Precision (FP16), translates using native generation, and unloads.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"⏳ Loading local NLLB model into VRAM (FP16 Mode) for {src_lang}...")
    
    # Initialize tokenizer with the specific source language
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    
    # Load model in Half-Precision to save VRAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Set the target language token ID for English
    tgt_lang = "eng_Latn"
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    paragraphs = text.split('\n')
    translated_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) > 0:
            # Tokenize the input text and move it to the GPU
            inputs = tokenizer(para, return_tensors="pt").to(device)
            
            # Generate the translation
            outputs = model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_token_id, 
                max_length=512
            )
            
            # Decode the generated tokens back into a readable string
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_paragraphs.append(translated_text)
            
            # Print a little progress indicator in the terminal
            print(f"  - Translated paragraph {i+1}/{len(paragraphs)}")
        else:
            translated_paragraphs.append("")
            
    # --- VRAM CLEANUP ---
    print("🧹 Translation complete. Unloading NLLB to secure space for Embeddings...")
    
    # Delete the large objects
    del model
    del tokenizer
    
    # Force Python to garbage collect
    gc.collect()
    
    # Force PyTorch to release the VRAM back to the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return "\n".join(translated_paragraphs)