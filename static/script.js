// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadForm = document.getElementById('upload-form');
const selectedFilesContainer = document.querySelector('.selected-files');
const uploadButton = document.querySelector('.upload-button');
const chatMessages = document.querySelector('.chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const loadingElement = document.querySelector('.loading');
const themeToggle = document.querySelector('.theme-toggle');
const sunIcon = document.querySelector('.sun-icon');
const moonIcon = document.querySelector('.moon-icon');

// Global variables
let currentSessionId = null;
let selectedFilesList = [];
let currentTheme = localStorage.getItem('theme') || 'light';

// Theme handling
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    if (theme === 'dark') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'light';
setTheme(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
});

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
uploadForm.addEventListener('submit', handleUpload);
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// File handling functions
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    updateSelectedFiles(files);
}

function updateSelectedFiles(files) {
    selectedFilesList = files;
    selectedFilesContainer.innerHTML = '';
    
    if (files.length > 0) {
        files.forEach(file => {
            const fileElement = document.createElement('div');
            fileElement.className = 'selected-file';
            fileElement.textContent = file.name;
            selectedFilesContainer.appendChild(fileElement);
        });
        uploadButton.disabled = false;
    } else {
        uploadButton.disabled = true;
    }
}

async function handleUpload(e) {
    e.preventDefault();
    
    if (selectedFilesList.length === 0) {
        showMessage('Please select at least one file to upload.', 'error');
        return;
    }
    
    // Show loading indicator
    loadingElement.style.display = 'block';
    uploadButton.disabled = true;
    
    try {
        const formData = new FormData();
        selectedFilesList.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to upload documents');
        }
        
        currentSessionId = data.session_id;
        showMessage('Documents uploaded successfully! You can now start chatting.', 'success');
        
        // Clear the file input and selected files
        fileInput.value = '';
        selectedFilesList = [];
        updateSelectedFiles([]);
        
    } catch (error) {
        console.error('Upload error:', error);
        showMessage(error.message, 'error');
    } finally {
        loadingElement.style.display = 'none';
        uploadButton.disabled = false;
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    if (!currentSessionId) {
        showMessage('Please upload documents first before chatting.', 'error');
        return;
    }
    
    // Add user message to chat
    addMessage(message, true);
    messageInput.value = '';
    
    // Show loading indicator
    loadingElement.style.display = 'block';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get response');
        }
        
        // Add bot response to chat
        addMessage(data.response, false);
        
        // Sources are still received but not displayed in chat
        
    } catch (error) {
        console.error('Chat error:', error);
        addMessage(`Error: ${error.message}`, 'error');
    } finally {
        loadingElement.style.display = 'none';
    }
}

function addMessage(message, isUser = false, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = message;
    messageDiv.appendChild(textDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong><br>' + sources.join('<br>');
        messageDiv.appendChild(sourcesDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showMessage(message, type = 'info') {
    addMessage(message, false);
}

function displayWelcomeMessage() {
    const welcomeMessage = {
        type: 'bot',
        content: 'Hello! I\'m your academic assistant. You can upload documents (PDF, Word, Text, or Images) and ask me questions about them.',
        sources: []
    };
    appendMessage(welcomeMessage);
}

// Update the appendMessage function to handle sources better
function appendMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message.content;
    messageDiv.appendChild(contentDiv);

    if (message.sources && message.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        const sourcesList = document.createElement('ul');
        message.sources.forEach(source => {
            const sourceItem = document.createElement('li');
            sourceItem.textContent = `Source: ${source}`;
            sourcesList.appendChild(sourceItem);
        });
        sourcesDiv.appendChild(sourcesList);
        messageDiv.appendChild(sourcesDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}