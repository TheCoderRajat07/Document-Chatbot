# Document Chatbot

A powerful document-based chatbot that allows you to upload documents and ask questions about their content. Built with LangChain, FAISS, and Google's Gemini AI.

## Features

- Upload multiple document types (PDF, DOCX, DOC, images)
- Drag-and-drop file upload interface
- Real-time chat interface with document context
- Source attribution for answers
- Modern, responsive UI
- Session management for document processing

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: LangChain, FAISS, Google Gemini AI
- **Document Processing**: PyPDF, python-docx, Unstructured

## Setup and Installation

### Prerequisites

- Python 3.8+
- Google API Key for Gemini AI
- Tesseract OCR (for image processing)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/document-chatbot.git
   cd document-chatbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload one or more documents using the drag-and-drop interface or file browser
2. Click "Process Documents" to analyze the content
3. Once processing is complete, you'll be taken to the chat interface
4. Ask questions about the content of your documents
5. View the sources of information in the right panel

## Deployment

### Local Deployment

For local deployment, simply run the Flask application:

```
python app.py
```

### GitHub Pages Deployment

The project is set up for GitHub Pages deployment. The root `index.html` file redirects to the static version of the application.

1. Push your code to a GitHub repository
2. Enable GitHub Pages in the repository settings
3. Select the branch to deploy (usually `main` or `master`)

### Production Deployment

For production deployment, it's recommended to use a WSGI server like Gunicorn:

```
gunicorn app:app
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Google Gemini AI](https://ai.google.dev/) for the language model
- [Flask](https://flask.palletsprojects.com/) for the web framework 