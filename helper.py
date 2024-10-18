from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.documents import Document
import time
import os
from pinecone import Pinecone, ServerlessSpec
import shutil
def remove_files_from_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all the entries in the directory
    for entry in os.listdir(folder_path):
        # Create full path
        file_path = os.path.join(folder_path, entry)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Remove the file
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Remove the directory and all its contents
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def load_index(api_key_pinecone):
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = api_key_pinecone

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "langchain-docs-index"  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
            

    index = pc.Index(index_name)
    
    
    return index

def load_vector_store(api_key_google,api_key_pinecone):
    os.environ['GOOGLE_API_KEY'] = api_key_google
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    index = load_index(api_key_pinecone)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    
    return vector_store

def add_docs_to_index(vector_store):
    loader = PyPDFDirectoryLoader("docs")
    data = loader.load_and_split()
    context = "\n".join(str(p.page_content) for p in data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in data)
    texts = text_splitter.split_text(context)
    documents = [Document(page_content=text) for text in texts]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return uuids
    
def get_answer(question, vector_store):
    results = vector_store.similarity_search(
    question,
    k=2,
    )
    prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain(
    {"input_documents":results, "question": question}
    , return_only_outputs=True)
    
    return response['output_text'], [res.page_content for res in results]