# You can use the Hugging Face Inference API for your demo
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/Raazi29/Nyaya-Llama-3.1-8B-Indian-Legal"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

def scrutinize_legal_data(facts, evidence, ipc):
    prompt = f"""
    ### Instruction:
    As an Indian Legal Expert, scrutinize the following Evidence against the provided Facts. 
    Identify any inconsistencies or missing legal requirements from both appelant and defendant.

    ### Facts:
    {facts}

    ### Evidence:
    {evidence}

    ### IPC/STATUTES:
    {ipc}

    ### Response:
    """
    
    # Send the extracted data to Nyaya-Llama
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()[0]['generated_text']