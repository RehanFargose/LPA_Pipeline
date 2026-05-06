from google import genai
import streamlit as st

def stream_wrapper(response):
    """Helper to yield text chunks from the new SDK's stream response."""
    for chunk in response:
        yield chunk.text


# 1. Create a global variable to hold the client
_global_client = None


def get_gemini_client():
    """
    Initializes and returns the modern Gemini API client.
    The API key is retrieved from .streamlit/secrets.toml.
    """
    
    """Initializes the client ONCE and keeps it alive."""
    global _global_client
    if _global_client is None:
        # Initialize only if it doesn't exist yet
        _global_client = genai.Client(api_key=st.secrets["GEMINI_LPA_KEY"])
        
        
    generate_models = [m.name for m in _global_client.models.list() 
                   if 'generateContent' in m.supported_actions]
    print("Models you can actually use for generation:")
    print(generate_models)
        
    return _global_client



def prompt_1_document_synthesis(raw_text):
    """Stage 1: Synthesize raw case documents into structured formats."""
    curr_model = "gemini-2.5-flash"
    
    client = get_gemini_client()
    
    # Use the client.models.generate_content method
    response = client.models.generate_content_stream(
        model=curr_model,
        contents=f"Extract and structure the core facts, parties involved, and primary dispute from this raw legal text:\n\n{raw_text[:8000]}",
    )
    return stream_wrapper(response)
    


def prompt_2_evidence_scrutiny(structured_context):
    """Stage 2: Scrutiny of evidence against facts."""
    curr_model = "gemini-2.5-flash"
    
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=curr_model,
        contents=f"Based on these structured facts and arguments, scrutinize the provided evidence for consistency and legal relevance:\n\n{structured_context}",
    )
    return stream_wrapper(response)
    
    


def prompt_3_precedent_analysis(evidence_analysis, precedents):
    """Stage 3: Identify relevant Indian case law and align with precedents."""
    curr_model = "gemini-2.5-flash"
    precedent_text = "\n".join([p['document'] for p in precedents[:5]])

    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=curr_model,
        contents=f"Compare this evidence analysis with these Supreme Court precedents (1990-2025). Identify direct overlaps:\n\nAnalysis: {evidence_analysis}\n\nPrecedents: {precedent_text}",
    )
    return stream_wrapper(response)
    

def prompt_4_verdict_prediction(aligned_precedents):
    """Stage 4: Generate reasoned outcomes based on statutory alignment."""
    curr_model = "gemini-2.5-flash"
    
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=curr_model,
        contents=f"Predict a probable legal outcome or 'verdict' based on statutory alignment and the preceding analysis:\n\n{aligned_precedents}",
    )
    return stream_wrapper(response)
    


def prompt_5_executive_summary(final_prediction):
    """Stage 5: Create executive case reports for non-lawyers."""
    curr_model = "gemini-2.5-flash"
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=curr_model,
        contents=f"Create a short, simplified executive summary of this legal analysis for a non-lawyer/client:\n\n{final_prediction}",
    )
    return stream_wrapper(response)