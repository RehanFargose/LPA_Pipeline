import streamlit as st
from RAG_Stages.legal_utils import process_document

def perform_precedent_search(uploaded_files, model, collection):
    """
    Handles document processing, chunking, and ChromaDB querying.
    Returns a sorted list of results and the combined cleaned text.
    """
    all_results_map = {}
    all_cleaned_texts = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            cleaned_text = process_document(uploaded_file)
            if not cleaned_text:
                continue
            
            all_cleaned_texts.append(cleaned_text)
            
            # Chunking logic
            WINDOW_SIZE = 1000
            OVERLAP = 200
            file_chunks = [cleaned_text[i:i + WINDOW_SIZE] 
                        for i in range(0, len(cleaned_text), WINDOW_SIZE - OVERLAP)]
            
            # Sampling first 5 chunks
            sampled_chunks = file_chunks[:5] 
            
            for chunk in sampled_chunks:
                query_vector = model.encode(chunk).tolist()
                chunk_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=10,
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

    sorted_results = sorted(all_results_map.values(), key=lambda x: (x['distance'], -x['frequency']))
    combined_text = "\n\n".join(all_cleaned_texts)
    
    return sorted_results, combined_text