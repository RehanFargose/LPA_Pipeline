import streamlit as st
import chromadb

# Configure the page
st.set_page_config(page_title="Document Viewer", page_icon="📄", layout="wide")

# Retrieve the document ID from the URL query parameters
query_params = st.query_params
if "doc_id" in query_params:
    doc_id = query_params["doc_id"]
    
    # Re-initialize DB connection
    client = chromadb.PersistentClient(path="C:/LPA_Vector_DB/chroma_persistent_storage")
    collection = client.get_collection(name="legal_precedents")
    
    # Step 1: Fetch the specific chunk to get its metadata (like case_no)
    initial_result = collection.get(ids=[doc_id])
    
    if initial_result and initial_result['documents']:
        metadata = initial_result['metadatas'][0]
        case_name = metadata.get('case_name', 'Unknown Case')
        target_case_no = metadata.get('case_no')
        
        # Step 2: Fetch ALL chunks for this specific case and stitch them together
        if target_case_no:
            # Query ChromaDB for all chunks with the same case number
            all_chunks = collection.get(where={"case_no": target_case_no})
            
            # Helper function to sort chunks correctly (e.g., c1, c2 ... c10)
            # This relies on your chunk IDs ending in something like "_c24"
            def get_chunk_index(chunk_id):
                try:
                    return int(chunk_id.split('_c')[-1])
                except ValueError:
                    return 0 # Fallback if parsing fails
            
            # Zip IDs and documents, sort them by chunk index, and extract just the text
            sorted_docs = sorted(
                zip(all_chunks['ids'], all_chunks['documents']), 
                key=lambda x: get_chunk_index(x[0])
            )
            
            # Join all sorted chunks with a double newline for readability
            full_text = "\n\n".join([doc_text for _, doc_text in sorted_docs])
            total_chunks_found = len(sorted_docs)
        else:
            # Fallback if there is no case_no to group by
            full_text = initial_result['documents'][0]
            total_chunks_found = 1

        # --- Top Level ---
        st.title(f"⚖️ {case_name}")
        st.markdown("---")
        
        # --- Layout setup: Left column (Metadata), Right column (Text) ---
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("Metadata")
            st.write(f"📅 **Year:** {metadata.get('year', 'N/A')}")
            st.write(f"🆔 **Matched Chunk ID:** `{doc_id}`")                    
            st.write(f"📅 **Case No:** {metadata.get('case_no', 'N/A')}")
            st.write(f"📅 **Acts:** {metadata.get('acts', 'N/A')}")
            st.write(f"📅 **Coram:** {metadata.get('coram', 'N/A')}")
            st.write(f"📅 **Decision_Date:** {metadata.get('decision_date', 'N/A')}")
            st.write(f"📅 **Disposal Nature:** {metadata.get('disposal_nature', 'N/A')}")
            st.write(f"📅 **Neutral Citation:** {metadata.get('neutral_citation', 'N/A')}")
            st.write(f"📅 **Precedents:** {metadata.get('precedents', 'N/A')}")
            
            st.divider()
            st.info(f"**Document Reconstructed:**\nStitched together {total_chunks_found} chunks to form this full text.")
            
        with col2:
            st.subheader("Full Document Text")
            # Using a container with some height can make reading very long documents easier
            with st.container(height=800, border=True):
                st.write(full_text)
            
    else:
        st.error("Document not found in the database.")
else:
    st.warning("No document ID provided. Please open a document from the main search page.")