# PDF RAG App Documentation (Rishit Chugh Sample Set Submission)
Note that the documentation for the ipynb is in the nb itself.
## Table of Contents
1. [Introduction](#introduction)
2. [Running the Streamlit App](#running-the-streamlit-app)
3. [Running with Docker](#running-with-docker)
4. [Using the App](#using-the-app)
5. [Developer Section](#developer-section)

## Introduction

This Streamlit app allows users to upload PDF files, ask questions about their content, and receive answers based on the information in the PDFs. The app uses language models and vector stores to process and retrieve relevant information.

## Running with Docker

To run the application using Docker:

1. Ensure you have Docker installed on your system.
2. Open a terminal or command prompt.
3. Navigate to the directory containing  `Dockerfile` and application files.
4. Build the Docker image:

```
docker build -t pdf-qa-app .
```

5. Run the Docker container:

```
docker run -p 8501:8501 pdf-qa-app
```

6. Open a web browser and go to `http://localhost:8501` to access the application.

## Running the Streamlit App (without docker)

To run the Streamlit app directly:

1. Ensure you have all the required dependencies installed.
2. Open a terminal or command prompt.
3. Navigate to the directory containing `app.py`.
4. Run the following command:

```
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.






## Using the App

1. **Upload PDFs**: 
   - add api_keys for google gemini and pinecone
   - Click on the "Upload PDF files" button.
   - Select one or more PDF files from your computer.
   - The app will display the number of files uploaded.

2. **Ask a Question**:
   - Once the PDFs are uploaded, you'll see a text input field.
   - Type your question related to the content of the uploaded PDFs.
   - Press Enter or click outside the input field to submit the question.

3. **View Results**:
   - The app will process your question and display two sections:
     - "Answer from LLM": The generated answer to your question.
     - "Retrieved Documents": Relevant text snippets from the PDFs used to answer the question.

4. **Clear Uploaded Files**:
   - To remove the uploaded files and clear the vector database, click the "Quit VectorDB" button.
   - You'll see a success message when the files are cleared.

## Developer Section

### Main App (app.py)

- `UPLOAD_FOLDER`: Defines the directory where uploaded PDFs are stored.
- `uploaded_files`: Handles multiple PDF file uploads.
- `vector_store`: Loads or creates a vector store for document indexing.
- `uuids`: Stores unique identifiers for added documents.
- `get_answer()`: Processes the user's question and retrieves the answer.

### Helper Functions (helper.py)

1. `remove_files_from_folder(folder_path)`:
   - Removes all files and subdirectories from the specified folder.

2. `load_index()`:
   - Initializes and returns a Pinecone index for vector storage.

3. `load_vector_store()`:
   - Creates a PineconeVectorStore with Google AI embeddings.

4. `add_docs_to_index(vector_store)`:
   - Loads PDFs, splits them into chunks, and adds them to the vector store.
   - Returns UUIDs for the added documents.

5. `get_answer(question, vector_store)`:
   - Performs similarity search on the vector store.
   - Uses a language model to generate an answer based on the retrieved context.
   - Returns the answer and relevant document snippets.

These functions work together to process PDFs, index their content, and provide answers to user queries using advanced language models and vector search techniques.

# Design Choices

1. Langchain was used due to its orchestrating capabilities with multiple different services and fortunately it supports pinecone aswell

2. Helper functions were seperated because of modularity and clean code

3. Gemini was used since its a free API

4. Challenges weren't faced as such, but currently the vector db needs to be deleted otherwise it will get populated again and again without getting rid of the data from the previous uses. there is a better implementation that I didn't implement.