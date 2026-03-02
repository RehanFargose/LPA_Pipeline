import requests
import os

# New Router URL format
API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"
# Make sure your HF_TOKEN is set in your environment variables
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

def scrutinize_legal_data(facts, evidence, ipc):
    # Use the Chat Completion format for better results with Llama 3.1
    prompt = f"Scrutinize this evidence against facts: {facts}\nEvidence: {evidence}\nIPC: {ipc}"
    
    payload = {
        "model": "Raazi29/Nyaya-Llama-3.1-8B-Indian-Legal",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
   # Check if the server actually sent content
    if response.status_code != 200:
        print(f"Server returned error {response.status_code}: {response.text}")
        return f"Error: Server returned status {response.status_code}"
        
    if not response.text.strip():
        return "Error: Received an empty response from Hugging Face. The model might still be loading."

    try:
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return f"Unexpected JSON structure: {result}"
    except requests.exceptions.JSONDecodeError:
        return f"Failed to decode JSON. Raw response: {response.text}"