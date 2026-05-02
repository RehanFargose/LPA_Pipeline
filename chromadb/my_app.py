import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import zipfile
import os

def prepare_database(zip_path, extract_to="./my_legal_db"):
    # Only extract if the database directory doesn't already exist
    if not os.path.exists(extract_to):
        with st.spinner("Extracting 1.48GB Legal Database..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        st.success("Database Ready!")

# Call this before loading resources
# Ensure 'legal_index_package.zip' is in the same folder as your app
prepare_database("legal_index_package.zip")


# --- Page Configuration ---
st.set_page_config(page_title="Legal NLP Explorer", layout="wide")
st.title("⚖️ Indian Supreme Court Precedent Explorer")
st.markdown("### MTech Thesis Research: 1990–2025 Judgment Analysis")

# --- Initialize Resources ---
@st.cache_resource
def load_chroma():
    # Connect to the persistent path you set up earlier
    client = chromadb.PersistentClient(path="./my_legal_db")
    return client.get_or_create_collection(name="sc_judgments")

@st.cache_resource
def load_model():
    # Using a lightweight model suitable for GTX 1650
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

collection = load_chroma()
model = load_model()

# --- Sidebar: Database Stats ---
with st.sidebar:
    st.header("Database Overview")
    total_docs = collection.count()
    st.metric("Total Chunks Indexed", total_docs)
    st.info("Dataset scope: 1990 - 2025 Indian SC Judgments")

# --- Main UI: Search Interface ---
query_text = st.text_input("Enter a legal concept or fact pattern (e.g., 'Environmental negligence in Maharashtra'):")

if query_text:
    with st.spinner("Searching vector space..."):
        # 1. Embed the query text
        query_vector = model.encode(query_text).tolist()

        # 2. Query ChromaDB for top 3 results
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

    # --- Displaying the "Compass-like" Results ---
    st.subheader("Top Matching Precedents")
    
    for i in range(len(results['documents'][0])):
        with st.expander(f"Result {i+1}: {results['metadatas'][0][i].get('case_name', 'Unknown Case')}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Metadata**")
                st.write(f"📅 **Year:** {results['metadatas'][0][i].get('year', 'N/A')}")
                st.write(f"📏 **Distance:** {round(results['distances'][0][i], 4)}")
            
            with col2:
                st.write("**Excerpt (Retrieved Chunk)**")
                st.write(results['documents'][0][i])

# --- Placeholder for Gemini Integration ---
st.divider()
if st.button("Analyze with Gemini"):
    st.warning("Next step: Connect your Google API key to synthesize these results!")
    
    
    
# To run the app:    streamlit run my_app.py
# T0 check GPU usage: nvitop 
