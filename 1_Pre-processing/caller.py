import os
from dotenv import load_dotenv
from google import genai
import json
from google.genai import types # The new unified types

load_dotenv()

# Initialize the new GenAI Client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def extract_legal_details(legal_text):
    prompt = f"""
    You are a Legal Expert in Indian Law. Analyze the following court text and extract:
    1. FACTS: The core background of the dispute.
    2. EVIDENCE: Specific documents, testimonies, or proofs cited.
    3. IPC/STATUTES: Any Indian Penal Code sections or specific Acts mentioned.

    Format the output as a clean Python Dictionary.
    
    TEXT:
    {legal_text}
    """
    
    # Use the new models.generate_content method
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
            response_mime_type="application/json",
        )
    )


    try:
        data = json.loads(response.text)
        return data
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return None