import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json
from RAG_Stages.legal_utils import process_document




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





# --- NEW UI SECTION ---
st.header("⚖️ Dispute Resolution & Mediation Portal")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload all relevant documents (Arguments, Evidence, Fact Sheets)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)


if st.button("Find Precedents"):
    if uploaded_files:
        all_cleaned_text = []
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                cleaned_text = process_document(uploaded_file)
                if cleaned_text:
                    all_cleaned_text.append(cleaned_text)
            full_case_context = "\n\n".join(all_cleaned_text)
            
        with st.spinner("Analyzing case segments and searching precedents..."):
            # --- Sliding Window Implementation ---
            WINDOW_SIZE = 1000  # Characters per chunk
            OVERLAP = 200       # Overlap to maintain context between windows
            
            chunks = [full_case_context[i:i + WINDOW_SIZE] 
                      for i in range(0, len(full_case_context), WINDOW_SIZE - OVERLAP)]
            
            # To avoid hitting performance bottlenecks, we limit the number of windows
            max_chunks = chunks[:10] 
            
            all_results_map = {} # Store unique results and their best (minimum) distance

            for chunk in max_chunks:
                query_vector = model.encode(chunk).tolist()
                chunk_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Aggregate results
                for i in range(len(chunk_results['ids'][0])):
                    res_id = chunk_results['ids'][0][i]
                    dist = chunk_results['distances'][0][i]
                    
                    if res_id not in all_results_map or dist < all_results_map[res_id]['distance']:
                        all_results_map[res_id] = {
                            "id": res_id,
                            "document": chunk_results['documents'][0][i],
                            "metadata": chunk_results['metadatas'][0][i],
                            "distance": dist,
                            "frequency": all_results_map.get(res_id, {}).get("frequency", 0) + 1
                        }
            
            # Sort by distance (ascending) then frequency (descending)
            sorted_results = sorted(
                all_results_map.values(), 
                key=lambda x: (x['distance'], -x['frequency'])
            )

        # --- Displaying the Aggregated Results ---
        st.subheader(f"Top Matching Precedents (Analyzed {len(max_chunks)} segments)")
        
        for i, res in enumerate(sorted_results[:30]): # Show top 5 unique precedents
            with st.expander(f"Result {i+1}: {res['metadata'].get('case_name', 'Unknown Case')}"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.write("**Metadata**")
                    # Fixed metadata
                    st.write(f"📅 **Year:** {res['metadata'].get('year', 'N/A')}")
                    st.write(f"🆔 **Chunk/File ID:** `{res['id']}`")                    
                    st.write(f"📅 **Case No:** {res['metadata'].get('case_no', 'N/A')}")
                    st.write(f"📅 **Acts:** {res['metadata'].get('acts', 'N/A')}")
                    st.write(f"📅 **Coram:** {res['metadata'].get('coram', 'N/A')}")
                    st.write(f"📅 **Decision_Date:** {res['metadata'].get('decision_date', 'N/A')}")
                    st.write(f"📅 **Disposal Nature:** {res['metadata'].get('disposal_nature', 'N/A')}")
                    st.write(f"📅 **Neutral Citation:** {res['metadata'].get('neutral_citation', 'N/A')}")
                    
                    # Cosine similarity/distance, lower is better
                    st.write(f"📏 **Distance:** {round(res['distance'], 4)}")
                    
                    # Similar Segments retrieved
                    st.write(f"🔄 **Segment Hits:** {res['frequency']}")
                
                with col2:
                    st.write("**Excerpt**")
                    st.write(res['document'])
    else:
        st.error("Please upload at least one document.")





# --- Sidebar: Database Stats ---
with st.sidebar:
    st.header("Database Overview")
    total_docs = collection.count()
    st.metric("Total Chunks Indexed", total_docs)
    st.info("Dataset scope: 1990 - 2025 Indian SC Judgments")




    
# To run the app:    streamlit run my_app.py
# To check GPU usage: nvitop 
