import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json
from legal_utils import process_document


def retrieve_precedents(uploaded_files):
    
    if st.button("Find Precedents"):
        if uploaded_files:
            all_results_map = {}
            
            # --- Process Each File Individually ---
            for uploaded_file in uploaded_files:
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    cleaned_text = process_document(uploaded_file)
                    if not cleaned_text:
                        continue
                    
                    # Create sliding windows for THIS specific file
                    WINDOW_SIZE = 1000
                    OVERLAP = 200
                    file_chunks = [cleaned_text[i:i + WINDOW_SIZE] 
                                for i in range(0, len(cleaned_text), WINDOW_SIZE - OVERLAP)]
                    
                    # Take the most important parts of THIS document (e.g., first 5 chunks)
                    # This ensures the 2nd and 3rd docs ALWAYS get a search slot
                    sampled_chunks = file_chunks[:5] 
                    
                    for chunk in sampled_chunks:
                        query_vector = model.encode(chunk).tolist()
                        chunk_results = collection.query(
                            query_embeddings=[query_vector],
                            n_results=30,
                            include=["documents", "metadatas", "distances"]
                        )
                        
                        # Aggregate results into the global map
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
            # --- Displaying the Aggregated Results ---
            # st.subheader(f"Top Matching Precedents (Analyzed {len(max_chunks)} segments)")
            
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
                        st.write(f"📅 **Precedents:** {res['metadata'].get('precedents', 'N/A')}")

                        # Cosine similarity/distance, lower is better
                        st.write(f"📏 **Distance:** {round(res['distance'], 4)}")
                        
                        # Similar Segments retrieved
                        st.write(f"🔄 **Segment Hits:** {res['frequency']}")
                    
                    with col2:
                        st.write("**Excerpt**")
                        st.write(res['document'])
        else:
            st.error("Please upload at least one document.")

