# app/ui/app.py
import streamlit as st
import requests
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Document Understanding", layout="centered")

st.title("AI Document Understanding System")
st.write("Upload documents and ask questions with multi-document support and confidence-based refusal.")

st.sidebar.header("Document Library")

try:
    response = requests.get(f"{API_BASE}/documents")
    if response.status_code == 200:
        docs_data = response.json()
        documents = docs_data["documents"]
        total_docs = docs_data["total_documents"]
        total_chunks = docs_data["total_chunks"]
        
        st.sidebar.metric("Total Documents", total_docs)
        st.sidebar.metric("Total Chunks", total_chunks)
        
        if documents:
            st.sidebar.subheader("Uploaded Documents")
            for doc in documents:
                with st.sidebar.expander(f"{doc['filename'][:30]}..."):
                    st.write(f"ID: {doc['document_id']}")
                    st.write(f"Chunks: {doc['chunks_count']}")
                    st.write(f"Uploaded: {doc.get('upload_timestamp', 'N/A')[:19]}")
                    
                    if st.button(f"Delete", key=f"delete_{doc['document_id']}"):
                        del_response = requests.delete(f"{API_BASE}/documents/{doc['document_id']}")
                        if del_response.status_code == 200:
                            st.success("Document deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
        else:
            st.sidebar.info("No documents uploaded yet")
    else:
        st.sidebar.error("Cannot connect to API")
except Exception as e:
    st.sidebar.error(f"API Error: {str(e)}")

st.sidebar.divider()

st.header("Upload Document")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"Selected: {uploaded_file.name}")
        st.write(f"Size: {uploaded_file.size / 1024:.2f} KB")
    
    with col2:
        if st.button("Upload", type="primary", use_container_width=True):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    }
                    response = requests.post(f"{API_BASE}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Document uploaded successfully")
                        st.info(f"Document ID: {result['document_id']}")
                        st.info(f"Chunks created: {result['chunks_created']}")
                        st.rerun()
                    else:
                        error = response.json()
                        st.error(f"Upload failed: {error.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.divider()

st.header("Ask a Question")

try:
    response = requests.get(f"{API_BASE}/documents")
    if response.status_code == 200:
        docs_data = response.json()
        documents = docs_data["documents"]
        
        if documents:
            doc_options = {f"{doc['filename']} ({doc['document_id'][:8]}...)": doc['document_id'] 
                          for doc in documents}
            
            selected_doc_label = st.selectbox(
                "Select document to query",
                options=list(doc_options.keys())
            )
            
            selected_doc_id = doc_options[selected_doc_label]
            
            question = st.text_area(
                "Enter your question",
                placeholder="What is the main topic of this document?"
            )
            
            if st.button("Ask Question", type="primary"):
                if not question or len(question.strip()) < 10:
                    st.warning("Question must be at least 10 characters")
                elif len(question) > 1000:
                    st.warning("Question must be less than 1000 characters")
                else:
                    with st.spinner("Thinking..."):
                        try:
                            payload = {
                                "document_id": selected_doc_id,
                                "question": question.strip()
                            }
                            response = requests.post(f"{API_BASE}/ask", json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                if result["refused"]:
                                    st.warning("Question Refused (Low Confidence)")
                                    st.write(result["answer"])
                                    
                                    with st.expander("Why was this refused?"):
                                        st.write(f"Confidence Score: {result['confidence_score']:.2%}")
                                        st.write(f"Threshold Required: 65%")
                                        st.write(f"Reasoning: {result.get('reasoning', 'Below confidence threshold')}")
                                else:
                                    st.success("Answer (High Confidence)")
                                    st.markdown(f"### {result['answer']}")
                                    
                                    with st.expander("Response Details"):
                                        col1, col2 = st.columns(2)
                                        col1.metric("Confidence", f"{result['confidence_score']:.2%}")
                                        col2.metric("Sources Used", result['sources_used'])
                            
                            elif response.status_code == 404:
                                st.error("Document not found")
                            else:
                                error = response.json()
                                st.error(f"Error: {error.get('detail', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        else:
            st.info("Please upload a document first")
    
    else:
        st.error("Cannot connect to API")

except Exception as e:
    st.error(f"API Error: {str(e)}")

st.divider()
st.caption("Multi-document support, similarity-based refusal, comprehensive testing")

