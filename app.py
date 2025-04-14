import os
import sys
import warnings
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import uuid

# Langchain components
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module='unstructured')

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with GOOGLE_API_KEY=YOUR_KEY")
    sys.exit(1)

MODEL_NAME = "gemini-1.5-pro"  # Updated to use the correct model name
EMBEDDING_MODEL_NAME = "models/embedding-001"  # Standard Google embedding model

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Global variables to store session data
sessions = {}

# --- Document Loading and Processing ---

def load_document(file_path: str) -> Optional[List[Document]]:
    """Loads a single document based on its extension."""
    _, file_extension = os.path.splitext(file_path.lower())
    loader = None

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            # Unstructured handles both .doc and .docx
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            # Unstructured handles images and performs OCR
            # Ensure Tesseract is installed and in PATH
            loader = UnstructuredImageLoader(file_path, mode="single")  # "single" or "elements"
        else:
            # Attempt generic loading for other types if needed
            print(f"Warning: Unsupported file type '{file_extension}'. Attempting generic load.")
            loader = UnstructuredFileLoader(file_path, mode="single")  # Try generic loader
            if not loader:
                return None

        if loader:
            print(f"Loading document: {os.path.basename(file_path)}")
            return loader.load()
        else:
            return None

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ImportError as e:
        print(f"Error: Missing dependency for {file_extension}. {e}")
        print("Please ensure all required libraries (pypdf, python-docx, unstructured, Pillow, pytesseract) are installed.")
        return None
    except Exception as e:
        print(f"Error loading document {os.path.basename(file_path)}: {e}")
        # Specific error handling for OCR if Tesseract is not found
        if "tesseract is not installed or isn't in your PATH" in str(e).lower():
            print("\n*** Tesseract OCR Error ***")
            print("Tesseract is required for image processing but not found.")
            print("Please install Tesseract and ensure it's added to your system's PATH.")
            print("Installation guides:")
            print("  macOS: brew install tesseract")
            print("  Linux (Debian/Ubuntu): sudo apt-get update && sudo apt-get install tesseract-ocr libtesseract-dev")
            print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.")
            print("-" * 30)
        return None


def load_and_process_documents(file_paths: List[str]) -> List[Document]:
    """Loads multiple documents, processes them, and splits into chunks."""
    all_docs = []
    successful_loads = 0
    for file_path in file_paths:
        docs = load_document(file_path)
        if docs:
            all_docs.extend(docs)
            successful_loads += 1
        else:
            print(f"Skipping file due to loading errors: {os.path.basename(file_path)}")

    if not all_docs:
        print("Error: No documents were successfully loaded.")
        return []

    print(f"\nSuccessfully loaded {successful_loads} document(s).")
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Created {len(split_docs)} document chunks.")
    return split_docs


# --- Vector Store and Embeddings ---

def create_vector_store(documents: List[Document]) -> Optional[FAISS]:
    """Creates a FAISS vector store from document chunks."""
    if not documents:
        print("Error: Cannot create vector store with no documents.")
        return None

    print("Generating embeddings and creating FAISS vector store...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=API_KEY
        )
        vector_store = FAISS.from_documents(documents, embeddings)
        print("FAISS vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        if "api_key" in str(e).lower():
            print("Hint: Check if your GOOGLE_API_KEY is valid and has permissions.")
        return None

# --- Retrieval and Chat Chain ---

def setup_retrieval_chain(vector_store: FAISS):
    """Sets up the retrieval-augmented generation chain."""
    print("Setting up retrieval chain with Gemini model...")
    try:
        # Define the LLM
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=API_KEY,
            temperature=0.3,  # Adjust creativity (0.0 = deterministic, 1.0 = creative)
            convert_system_message_to_human=True  # Important for some models
        )

        # Define the prompt template
        # This tells the LLM how to use the retrieved context
        prompt_template = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Keep the answer concise and relevant to the documents.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Other options: "mmr", "similarity_score_threshold"
            search_kwargs={'k': 5}  # Retrieve top 5 relevant chunks
        )

        # Create the chain that combines documents (context) for the LLM
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # Create the main retrieval chain
        # This chain first retrieves documents, then passes them to combine_docs_chain
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        print("Retrieval chain ready.")
        return retrieval_chain

    except Exception as e:
        print(f"Error setting up retrieval chain: {e}")
        if "api_key" in str(e).lower():
            print("Hint: Check if your GOOGLE_API_KEY is valid and has Gemini API permissions.")
        return None


# --- API Routes ---

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create a temporary directory to store uploaded files
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for file in files:
        if file.filename:
            # Save the file to the temporary directory
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            file_paths.append(file_path)
    
    # Process the documents
    doc_chunks = load_and_process_documents(file_paths)
    if not doc_chunks:
        return jsonify({'error': 'Failed to process documents'}), 500
    
    # Create vector store
    vector_store = create_vector_store(doc_chunks)
    if not vector_store:
        return jsonify({'error': 'Failed to create vector store'}), 500
    
    # Setup retrieval chain
    qa_chain = setup_retrieval_chain(vector_store)
    if not qa_chain:
        return jsonify({'error': 'Failed to set up QA chain'}), 500
    
    # Store session data
    sessions[session_id] = {
        'qa_chain': qa_chain,
        'temp_dir': temp_dir,
        'file_paths': file_paths
    }
    
    return jsonify({
        'session_id': session_id,
        'message': f'Successfully processed {len(file_paths)} document(s)',
        'file_count': len(file_paths)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Check for either 'question' or 'message' in the request
    question = data.get('question') or data.get('message')
    session_id = data.get('session_id')
    
    if not question:
        return jsonify({'error': 'Missing question/message'}), 400
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session_data = sessions[session_id]
    qa_chain = session_data['qa_chain']
    
    try:
        # Invoke the retrieval chain
        result = qa_chain.invoke({"input": question})
        
        # Extract sources for context
        sources = []
        for i, doc in enumerate(result.get('context', [])):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')  # For PDFs
            sources.append({
                'index': i+1,
                'source': os.path.basename(source),
                'page': page
            })
        
        return jsonify({
            'response': result.get('answer', 'No answer generated'),
            'sources': sources
        })
    
    except Exception as e:
        print(f"Error during chat: {str(e)}")
        return jsonify({'error': f'Error during chat: {str(e)}'}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_session():
    data = request.json
    if not data or 'session_id' not in data:
        return jsonify({'error': 'Missing session_id'}), 400
    
    session_id = data['session_id']
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session_data = sessions[session_id]
    temp_dir = session_data['temp_dir']
    
    # Clean up temporary files
    try:
        for file_path in session_data['file_paths']:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error cleaning up session {session_id}: {e}")
    
    # Remove session data
    del sessions[session_id]
    
    return jsonify({'message': 'Session cleaned up successfully'})

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 