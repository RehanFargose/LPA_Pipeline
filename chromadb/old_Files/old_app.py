import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json


# --- Page Configuration ---
st.set_page_config(page_title="Legal NLP Explorer", layout="wide")
st.title("⚖️ Indian Supreme Court Precedent Explorer")
st.markdown("### MTech Thesis Research: 1990–2025 Judgment Analysis")

# --- Initialize Resources ---
@st.cache_resource
def load_chroma():
    #  Setup persistent client
    client = chromadb.PersistentClient(path="C:/LPA_Vector_DB/chroma_persistent_storage")
    collection = client.get_or_create_collection(name="legal_precedents", metadata={"hnsw:space": "cosine"})

    # Check if the collection is empty
    if collection.count() == 0:
        st.info("First-time setup: Indexing raw vectors into ChromaDB...")
        
        # Define paths to your extracted files[cite: 3]
        base_path = "C:/LPA_Vector_DB/my_legal_db"
        vectors_path = os.path.join(base_path, "sc_vectors.npy")
        payload_path = os.path.join(base_path, "sc_payload.json")

        if os.path.exists(vectors_path) and os.path.exists(payload_path):
            # Load raw data
            embeddings = np.load(vectors_path)
            with open(payload_path, "r") as f:
                payloads = json.load(f)

            # 3. Batch insert into ChromaDB[cite: 1]
            BATCH_SIZE = 5000
            total_batches = (len(payloads) + BATCH_SIZE - 1) // BATCH_SIZE
            
            
            # Create Streamlit progress elements
            progress_text = "Indexing progress: 0%"
            my_bar = st.progress(0, text=progress_text)
            
            for i in range(0, len(payloads), BATCH_SIZE):
                batch_end = i + BATCH_SIZE
                batch_payloads = payloads[i:batch_end]
                
                collection.add(
                    ids=[p['id'] for p in batch_payloads],
                    embeddings=embeddings[i:batch_end].tolist(),
                    documents=[p['text'] for p in batch_payloads],
                    metadatas=[p['metadata'] for p in batch_payloads]
                )
                
                # Update progress bar
                current_batch = (i // BATCH_SIZE) + 1
                percent_complete = int((current_batch / total_batches) * 100)
                my_bar.progress(current_batch / total_batches, text=f"Indexing progress: {percent_complete}% ({current_batch}/{total_batches} batches)")
                
            my_bar.empty() # Remove the bar once finished 
            st.success("Indexing complete!")
        else:
            st.error(f"Raw files not found in {base_path}. Please check your folder structure.")
            
    return collection



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
        # Embed query text
        query_vector = model.encode(query_text).tolist()

        # Query ChromaDB for relevant results
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
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
                st.write(f"📅 **Acts:** {results['metadatas'][0][i].get('acts', 'N/A')}")
                st.write(f"📅 **Coram:** {results['metadatas'][0][i].get('coram', 'N/A')}")
                st.write(f"📅 **Decision_Date:** {results['metadatas'][0][i].get('decision_date', 'N/A')}")
                st.write(f"📅 **Case No:** {results['metadatas'][0][i].get('case_no', 'N/A')}")
                st.write(f"📅 **Disposal Nature:** {results['metadatas'][0][i].get('disposal_nature', 'N/A')}")
                st.write(f"📅 **Neutral Citation:** {results['metadatas'][0][i].get('neutral_citation', 'N/A')}")
                st.write(f"🆔 **Chunk/File ID:** {results['ids'][0][i]}")

                # Cosine similarity/distance, lower is better
                st.write(f"📏 **Distance:** {round(results['distances'][0][i], 4)}")
                
            
            with col2:
                st.write("**Excerpt (Retrieved Chunk)**")
                st.write(results['documents'][0][i])

# --- Placeholder for Gemini Integration ---
st.divider()
if st.button("Analyze with Gemini"):
    st.warning("Next step: Connect your Google API key to synthesize these results!")
    
    
    
# To run the app:    streamlit run my_app.py
# To check GPU usage: nvitop 
