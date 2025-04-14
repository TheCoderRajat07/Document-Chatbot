import os
import sys
import warnings
from dotenv import load_dotenv
from typing import List, Optional

# Langchain components
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader, # Generic loader
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='unstructured')

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with GOOGLE_API_KEY=YOUR_KEY")
    sys.exit(1)

MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Or "gemini-pro" or other compatible models
EMBEDDING_MODEL_NAME = "models/embedding-001" # Standard Google embedding model

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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
            loader = UnstructuredImageLoader(file_path, mode="single") # "single" or "elements"
        else:
            # Attempt generic loading for other types if needed
             print(f"Warning: Unsupported file type '{file_extension}'. Attempting generic load.")
             # loader = UnstructuredFileLoader(file_path, mode="single") # Uncomment if you want to try generic
             return None # Or raise an error if preferred

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
            temperature=0.3, # Adjust creativity (0.0 = deterministic, 1.0 = creative)
            convert_system_message_to_human=True # Important for some models
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
            search_type="similarity", # Other options: "mmr", "similarity_score_threshold"
            search_kwargs={'k': 5} # Retrieve top 5 relevant chunks
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


# --- Main Chat Interface ---

def main():
    print("-" * 50)
    print(" Custom Document Chatbot using LangChain, FAISS & Gemini")
    print("-" * 50)

    # --- Step 1: Get File Paths ---
    print("\nEnter the full paths to the documents you want to load.")
    print("Separate multiple paths with commas (,)")
    print("Supported types: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP")
    file_paths_input = input("File paths: ").strip()

    if not file_paths_input:
        print("No file paths provided. Exiting.")
        sys.exit(1)

    file_paths = [path.strip() for path in file_paths_input.split(',')]
    valid_files = [path for path in file_paths if os.path.exists(path)]
    invalid_files = [path for path in file_paths if not os.path.exists(path)]

    if invalid_files:
        print("\nWarning: The following files were not found and will be skipped:")
        for f in invalid_files:
            print(f" - {f}")

    if not valid_files:
        print("\nError: No valid files found at the specified paths. Exiting.")
        sys.exit(1)

    # --- Step 2: Load and Process Documents ---
    print("\n--- Starting Document Processing ---")
    doc_chunks = load_and_process_documents(valid_files)
    if not doc_chunks:
        print("Failed to process documents. Exiting.")
        sys.exit(1)
    print("--- Document Processing Complete ---")

    # --- Step 3: Create Vector Store ---
    print("\n--- Creating Vector Store ---")
    vector_store = create_vector_store(doc_chunks)
    if not vector_store:
        print("Failed to create vector store. Exiting.")
        sys.exit(1)
    print("--- Vector Store Ready ---")

    # --- Step 4: Setup QA Chain ---
    print("\n--- Setting up QA Chain ---")
    qa_chain = setup_retrieval_chain(vector_store)
    if not qa_chain:
        print("Failed to set up QA chain. Exiting.")
        sys.exit(1)
    print("--- QA Chain Ready ---")

    # --- Step 5: Start Chat Loop ---
    print("\n--- Starting Chat Interface ---")
    print("Ask questions based on your uploaded documents.")
    print("Type 'quit', 'exit', or 'bye' to end the chat.")

    while True:
        try:
            user_question = input("\nYou: ")
            if user_question.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye!")
                break

            if not user_question.strip():
                continue

            # Invoke the retrieval chain
            print("Chatbot: Thinking...")
            result = qa_chain.invoke({"input": user_question})

            # Print the answer
            print("\nChatbot:", result['answer'])

            # Optional: Print retrieved context sources (for debugging)
            # print("\nSources used:")
            # for i, doc in enumerate(result['context']):
            #      source = doc.metadata.get('source', 'Unknown')
            #      page = doc.metadata.get('page', 'N/A') # For PDFs
            #      print(f"  {i+1}. Source: {os.path.basename(source)}, Page: {page}")
            # print("-" * 10)


        except Exception as e:
            print(f"\nAn error occurred during chat: {e}")
            print("Please try again or restart the application.")
            # Optional: Add more specific error handling for API timeouts, etc.

if __name__ == "__main__":
    main()