import streamlit as st
import os
from PyPDF2 import PdfReader
from helper import load_vector_store, add_docs_to_index, get_answer, remove_files_from_folder


# Create a folder to store uploaded PDFs
UPLOAD_FOLDER = "docs"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
remove_files_from_folder(UPLOAD_FOLDER)

# Streamlit UI
st.title("Ask from PDF (Rishit Chugh sample Set assignment)")
google_api_key = st.text_input("Enter gemini API key")
pinecone_api_key = st.text_input("Enter pinecone API key")

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write(f"{len(uploaded_files)} file(s) uploaded.")

    # Save the uploaded files to the folder
    pdf_paths = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        pdf_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.success(f"PDFs stored in '{UPLOAD_FOLDER}' folder.")
    vector_store = load_vector_store(google_api_key, pinecone_api_key)
    uuids = add_docs_to_index(vector_store)

    # Input a question from the user
    question = st.text_input("Enter your question")

    if question:
        st.write(f"Processing your question: '{question}'")
        
        # Show output of the two functions
        response, docs = get_answer(question, vector_store)
    

        st.subheader("Answer from LLM")
        st.write(response)

        st.subheader("Retrieved Documents")
        st.write(docs)
    
    if st.button("Quit VectorDB"):
        vector_store.delete(ids=[uuids])
        st.success("Uploaded files cleared.")
