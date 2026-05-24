import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_gemini_chain(prompt_template):
    """
    Creates a standard LangChain 'Chain': 
    Prompt -> Model -> String Output
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        api_key=st.secrets["GEMINI_LPA_KEY"],
        streaming=True
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # This is the "LCEL" (LangChain Expression Language) syntax
    return prompt | model | StrOutputParser()


def prompt_1_document_synthesis(raw_text):
    template = f"Extract and structure core facts, parties, and statutes from this raw legal text: {{text}}"
    chain = get_gemini_chain(template)
    return chain.stream({"text": raw_text[:8000]})


def prompt_2_evidence_scrutiny(evidence_context):
    # template = f"Scrutinize the provided evidence for consistency and legal relevance based on these facts: {{evidence}}"
    template = f"""
        Analyze the provided evidence for consistency. 
        Focus specifically on:
        1. Allegations of matrimonial misconduct (e.g., unchastity, desertion).
        2. Whether these allegations were substantiated or withdrawn.
        3. Logical inconsistencies between the pleadings and the subsequent witness testimony.
        Facts: {{evidence}}
        """
    chain = get_gemini_chain(template)
    return chain.stream({"evidence": evidence_context})


def prompt_3_precedent_analysis(evidence_analysis, precedents):
    """Stage 3: Compare analysis with precedents, now including crucial metadata context."""
    
    formatted_precedents = []
    
    # Loop through the top 5 precedents and format text + metadata
    for i, p in enumerate(precedents[:5]):
        meta = p.get('metadata', {})
        
        # Build a structured string for each precedent
        entry = (
            f"--- Precedent {i+1} ---\n"
            f"Case Name: {meta.get('case_name', 'Unknown')} ({meta.get('year', 'N/A')})\n"
            f"Acts/Statutes: {meta.get('acts', 'N/A')}\n"
            f"Disposal Nature (Outcome): {meta.get('disposal_nature', 'N/A')}\n"
            f"Relevant Excerpt:\n{p.get('document', '')}\n"
        )
        formatted_precedents.append(entry)
        
    # Combine them all into one large string
    precedent_text = "\n".join(formatted_precedents)
    
    # Updated template that instructs the LLM to actually use the metadata
    template = f"""
    Compare the provided evidence analysis with the following Supreme Court precedents. 
    Pay special attention to how the 'Disposal Nature' (the outcome) and 'Acts' of these past cases align with or differ from the current case facts.
    
    Precedents: 
    {precedent_text}
    
    Analysis: 
    {{analysed_evidence}}
    """
    
    chain = get_gemini_chain(template)
    return chain.stream({"analysed_evidence": evidence_analysis})


# ... follow the same pattern for stages 4 and 5
def prompt_4_verdict_prediction(aligned_precedents):
    """Stage 4: Generate reasoned outcomes based on statutory alignment."""
    
    # template = f"Predict a probable legal outcome or 'verdict' based on statutory alignment, applicable ipc/bns statutes and the preceding analysis:\n\n{{aligned_precedents}}"
    #    3. **Sentencing/Relief**: 
    #    - If criminal: Predict jail time or fine amounts based on statutory limits.
    #    - If civil/family: Predict specific relief (e.g., Alimony amount, specific performance, or damages).
    template = f"""
    Act as a presiding judge. Based on the statutory alignment and preceding analysis, provide:
    1. **Disposition**: State clearly if the petition is Allowed, Dismissed, or Partially Allowed.
    2. **Reasoning**: Briefly link the specific IPC/BNS sections to the facts.
    3. ***Sentencing/Relief**: If specific amounts (Alimony/Costs/Jail time) are not mentioned in the text, state the statutory principle for determining them rather than inventing a figure.
    Context: {{aligned_precedents}}
    """
    chain = get_gemini_chain(template)
    return chain.stream({"aligned_precedents": aligned_precedents})
    
    
    
def prompt_5_executive_summary(final_prediction):
    """Stage 5: Create executive case reports for non-lawyers."""
    template = f"Create a short, simplified executive summary of this legal analysis for a non-lawyer/client:\n\n{{final_prediction}}"
    chain = get_gemini_chain(template)
    return chain.stream({"final_prediction": final_prediction})
    






