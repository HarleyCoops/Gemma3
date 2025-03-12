"""
Gemma 3 Web Application

A Flask-based web application that demonstrates practical use cases of Gemma 3
using the Google GenAI SDK, with commented code for local JAX execution as an alternative.
"""

import os
import io
import base64
import threading
import queue
import time
from typing import List, Optional, Dict, Any
import json
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Google GenAI SDK with your API key
if API_KEY:
    genai.configure(api_key=API_KEY)

# Uncomment the following for local JAX execution (if needed)
"""
import gemma as gm
import jax.numpy as jnp

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
"""

# Initialize Flask app
app = Flask(__name__)

# Queue for model responses
response_queue = queue.Queue()

class GemmaClient:
    """Client for interacting with Gemma 3 models."""
    
    def __init__(self, use_vision: bool = False):
        """
        Initialize the Gemma client.
        
        Args:
            use_vision: Whether to use the multimodal vision model.
        """
        self.use_vision = use_vision
        
        # Initialize with Google GenAI SDK
        if use_vision:
            self.model_name = "gemma-3-vision-it"
        else:
            self.model_name = "gemma-3-8b-it"
        
        self.model = genai.GenerativeModel(model_name=self.model_name)
        self.chat_session = self.model.start_chat(history=[])
        
        # For local execution (commented out)
        """
        if use_vision:
            self.local_model = gm.nn.Gemma3Vision_4B()
            self.local_params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3VISION_4B_IT)
        else:
            self.local_model = gm.nn.Gemma3_4B()
            self.local_params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
        
        self.tokenizer = gm.Tokenizer()
        self.chat_sampler = gm.text.ChatSampler(
            model=self.local_model,
            params=self.local_params,
            tokenizer=self.tokenizer,
            multi_turn=True
        )
        """
    
    def send_message(self, message: str, image_data: Optional[str] = None) -> str:
        """
        Send a message to the model, optionally with an image.
        
        Args:
            message: The text message to send.
            image_data: Optional base64-encoded image data.
            
        Returns:
            The model's response.
        """
        try:
            if image_data and self.use_vision:
                # Process image data
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Send multimodal content
                response = self.model.generate_content([message, image])
                return response.text
            else:
                # Text-only message
                response = self.chat_session.send_message(message)
                return response.text
        except Exception as e:
            return f"Error: {str(e)}"
        
        # For local execution (commented out)
        """
        try:
            if image_data and self.use_vision:
                # Process image data
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Send multimodal content
                response = self.chat_sampler.send_message(message, images=[image])
                return response
            else:
                # Text-only message
                response = self.chat_sampler.send_message(message)
                return response
        except Exception as e:
            return f"Error: {str(e)}"
        """
    
    def reset(self):
        """Reset the conversation history."""
        self.chat_session = self.model.start_chat(history=[])
        
        # For local execution (commented out)
        """
        self.chat_sampler.reset()
        """

# Global clients
text_client = None
vision_client = None

def initialize_clients():
    """Initialize the Gemma clients."""
    global text_client, vision_client
    
    if not API_KEY:
        print("Warning: Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        print("You can get an API key from Google AI Studio: https://makersuite.google.com/")
        return False
    
    try:
        print("Initializing Gemma clients...")
        text_client = GemmaClient(use_vision=False)
        vision_client = GemmaClient(use_vision=True)
        print("Gemma clients initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing Gemma clients: {str(e)}")
        return False

def process_text_query(prompt: str, conversation_id: str):
    """Process a text query and put the response in the queue."""
    try:
        if not text_client:
            response_queue.put({
                "conversation_id": conversation_id,
                "response": "Error: Gemma client not initialized. Please check your API key.",
                "error": "Client not initialized"
            })
            return
        
        response = text_client.send_message(prompt)
        response_queue.put({
            "conversation_id": conversation_id,
            "response": response,
            "error": None
        })
    except Exception as e:
        response_queue.put({
            "conversation_id": conversation_id,
            "response": None,
            "error": str(e)
        })

def process_image_query(prompt: str, image_data: str, conversation_id: str):
    """Process an image query and put the response in the queue."""
    try:
        if not vision_client:
            response_queue.put({
                "conversation_id": conversation_id,
                "response": "Error: Gemma vision client not initialized. Please check your API key.",
                "error": "Client not initialized"
            })
            return
        
        response = vision_client.send_message(prompt, image_data)
        response_queue.put({
            "conversation_id": conversation_id,
            "response": response,
            "error": None
        })
    except Exception as e:
        response_queue.put({
            "conversation_id": conversation_id,
            "response": None,
            "error": str(e)
        })

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/text', methods=['POST'])
def text_query():
    """Handle text queries."""
    data = request.json
    prompt = data.get('prompt', '')
    conversation_id = data.get('conversation_id', 'default')
    
    # Start processing in a separate thread
    threading.Thread(target=process_text_query, args=(prompt, conversation_id)).start()
    
    return jsonify({"status": "processing", "conversation_id": conversation_id})

@app.route('/api/image', methods=['POST'])
def image_query():
    """Handle image queries."""
    data = request.json
    prompt = data.get('prompt', '')
    image_data = data.get('image', '')
    conversation_id = data.get('conversation_id', 'default')
    
    # Start processing in a separate thread
    threading.Thread(target=process_image_query, args=(prompt, image_data, conversation_id)).start()
    
    return jsonify({"status": "processing", "conversation_id": conversation_id})

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history."""
    data = request.json
    conversation_type = data.get('type', 'text')
    
    if conversation_type == 'text' and text_client:
        text_client.reset()
    elif conversation_type == 'image' and vision_client:
        vision_client.reset()
    
    return jsonify({"status": "reset"})

@app.route('/api/status', methods=['GET'])
def check_status():
    """Check if there's a response ready."""
    try:
        # Non-blocking get
        result = response_queue.get_nowait()
        return jsonify({
            "status": "ready",
            "conversation_id": result["conversation_id"],
            "response": result["response"],
            "error": result["error"]
        })
    except queue.Empty:
        return jsonify({"status": "processing"})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models."""
    try:
        models = genai.list_models()
        gemma_models = [model.name for model in models if "gemma" in model.name.lower()]
        return jsonify({"models": gemma_models})
    except Exception as e:
        return jsonify({"error": str(e)})

def create_templates_and_static():
    """Create necessary template and static files."""
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemma 3 Web Interface</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Gemma 3 Web Interface</h1>
            <p class="subtitle">Powered by Google GenAI SDK</p>
            <div class="api-status" id="api-status">Checking API status...</div>
            <div class="tabs">
                <button id="text-tab" class="tab-button active">Text Mode</button>
                <button id="image-tab" class="tab-button">Image Mode</button>
            </div>
        </header>
        
        <main>
            <div id="text-interface" class="interface active">
                <div id="text-conversation" class="conversation"></div>
                <div class="input-area">
                    <textarea id="text-input" placeholder="Enter your message..."></textarea>
                    <button id="text-send">Send</button>
                    <button id="text-reset">Reset Conversation</button>
                </div>
            </div>
            
            <div id="image-interface" class="interface">
                <div id="image-conversation" class="conversation"></div>
                <div class="image-preview-container">
                    <div id="image-preview"></div>
                    <input type="file" id="image-upload" accept="image/*">
                    <label for="image-upload" class="upload-label">Choose Image</label>
                </div>
                <div class="input-area">
                    <textarea id="image-prompt" placeholder="Enter your prompt about the image..."></textarea>
                    <button id="image-send" disabled>Send</button>
                    <button id="image-reset">Reset Conversation</button>
                </div>
            </div>
        </main>
        
        <div id="loading" class="loading hidden">
            <div class="spinner"></div>
            <p>Processing with Gemma 3...</p>
        </div>
        
        <footer>
            <p>This application uses Gemma 3 models through the Google GenAI SDK.</p>
            <p>Get your API key from <a href="https://makersuite.google.com/" target="_blank">Google AI Studio</a>.</p>
        </footer>
    </div>
    
    <script src="/static/js/script.js"></script>
</body>
</html>""")
    
    # Create CSS file
    with open('static/css/styles.css', 'w') as f:
        f.write("""* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 20px;
}

h1 {
    color: #2c3e50;
    margin-bottom: 10px;
}

.subtitle {
    color: #7f8c8d;
    margin-bottom: 15px;
}

.api-status {
    padding: 5px 10px;
    border-radius: 5px;
    display: inline-block;
    margin-bottom: 15px;
    font-size: 14px;
}

.api-status.success {
    background-color: #d4edda;
    color: #155724;
}

.api-status.error {
    background-color: #f8d7da;
    color: #721c24;
}

.tabs {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.tab-button {
    padding: 10px 20px;
    background-color: #e0e0e0;
    border: none;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

.tab-button:first-child {
    border-radius: 5px 0 0 5px;
}

.tab-button:last-child {
    border-radius: 0 5px 5px 0;
}

.tab-button.active {
    background-color: #3498db;
    color: white;
}

.interface {
    display: none;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    margin-bottom: 20px;
}

.interface.active {
    display: block;
}

.conversation {
    height: 400px;
    overflow-y: auto;
    padding: 20px;
    border-bottom: 1px solid #e0e0e0;
}

.message {
    margin-bottom: 15px;
    padding: 10px 15px;
    border-radius: 5px;
    max-width: 80%;
}

.user-message {
    background-color: #e1f5fe;
    margin-left: auto;
    text-align: right;
}

.bot-message {
    background-color: #f1f1f1;
    margin-right: auto;
}

.input-area {
    padding: 20px;
    display: flex;
    flex-wrap: wrap;
}

textarea {
    flex: 1;
    min-height: 80px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    resize: none;
    font-family: inherit;
    margin-right: 10px;
}

button {
    padding: 10px 20px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
    margin-right: 10px;
}

button:hover {
    background-color: #2980b9;
}

button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

.image-preview-container {
    padding: 20px;
    text-align: center;
    border-bottom: 1px solid #e0e0e0;
}

#image-preview {
    max-width: 100%;
    max-height: 300px;
    margin-bottom: 15px;
}

#image-preview img {
    max-width: 100%;
    max-height: 300px;
    border-radius: 5px;
}

input[type="file"] {
    display: none;
}

.upload-label {
    display: inline-block;
    padding: 10px 20px;
    background-color: #3498db;
    color: white;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.upload-label:hover {
    background-color: #2980b9;
}

.loading {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.loading.hidden {
    display: none;
}

.spinner {
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin-bottom: 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading p {
    color: white;
    font-size: 18px;
}

.image-container {
    text-align: center;
    margin-bottom: 15px;
}

.image-container img {
    max-width: 100%;
    max-height: 200px;
    border-radius: 5px;
}

pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: #f8f8f8;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}

code {
    font-family: 'Courier New', Courier, monospace;
}

footer {
    text-align: center;
    padding: 20px 0;
    color: #7f8c8d;
    font-size: 14px;
}

footer a {
    color: #3498db;
    text-decoration: none;
}

footer a:hover {
    text-decoration: underline;
}""")
    
    # Create JavaScript file
    with open('static/js/script.js', 'w') as f:
        f.write("""document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const apiStatus = document.getElementById('api-status');
    const textTab = document.getElementById('text-tab');
    const imageTab = document.getElementById('image-tab');
    const textInterface = document.getElementById('text-interface');
    const imageInterface = document.getElementById('image-interface');
    const textInput = document.getElementById('text-input');
    const textSend = document.getElementById('text-send');
    const textReset = document.getElementById('text-reset');
    const textConversation = document.getElementById('text-conversation');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const imagePrompt = document.getElementById('image-prompt');
    const imageSend = document.getElementById('image-send');
    const imageReset = document.getElementById('image-reset');
    const imageConversation = document.getElementById('image-conversation');
    const loading = document.getElementById('loading');
    
    // State variables
    let currentConversationId = Date.now().toString();
    let currentImageData = null;
    let isProcessing = false;
    let statusCheckInterval = null;
    
    // Check API status
    checkApiStatus();
    
    // Tab switching
    textTab.addEventListener('click', function() {
        textTab.classList.add('active');
        imageTab.classList.remove('active');
        textInterface.classList.add('active');
        imageInterface.classList.remove('active');
    });
    
    imageTab.addEventListener('click', function() {
        imageTab.classList.add('active');
        textTab.classList.remove('active');
        imageInterface.classList.add('active');
        textInterface.classList.remove('active');
    });
    
    // Text mode functionality
    textSend.addEventListener('click', function() {
        sendTextMessage();
    });
    
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });
    
    textReset.addEventListener('click', function() {
        resetConversation('text');
    });
    
    // Image mode functionality
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                currentImageData = event.target.result;
                imagePreview.innerHTML = `<img src="${currentImageData}" alt="Uploaded Image">`;
                imageSend.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });
    
    imageSend.addEventListener('click', function() {
        sendImageMessage();
    });
    
    imagePrompt.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !imageSend.disabled) {
            e.preventDefault();
            sendImageMessage();
        }
    });
    
    imageReset.addEventListener('click', function() {
        resetConversation('image');
    });
    
    // Helper functions
    function checkApiStatus() {
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    apiStatus.textContent = 'API Error: ' + data.error;
                    apiStatus.className = 'api-status error';
                } else if (data.models && data.models.length > 0) {
                    apiStatus.textContent = 'API Connected: ' + data.models.length + ' Gemma models available';
                    apiStatus.className = 'api-status success';
                } else {
                    apiStatus.textContent = 'API Connected: No Gemma models found';
                    apiStatus.className = 'api-status error';
                }
            })
            .catch(error => {
                apiStatus.textContent = 'API Connection Error';
                apiStatus.className = 'api-status error';
                console.error('Error checking API status:', error);
            });
    }
    
    function sendTextMessage() {
        const message = textInput.value.trim();
        if (message && !isProcessing) {
            addMessage(textConversation, message, 'user');
            textInput.value = '';
            
            isProcessing = true;
            loading.classList.remove('hidden');
            
            fetch('/api/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: message,
                    conversation_id: currentConversationId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'processing') {
                    startStatusCheck();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                isProcessing = false;
                loading.classList.add('hidden');
                addMessage(textConversation, 'An error occurred while sending your message.', 'bot');
            });
        }
    }
    
    function sendImageMessage() {
        const prompt = imagePrompt.value.trim();
        if (prompt && currentImageData && !isProcessing) {
            addMessage(imageConversation, prompt, 'user');
            addImageToConversation(imageConversation, currentImageData);
            imagePrompt.value = '';
            
            isProcessing = true;
            loading.classList.remove('hidden');
            
            fetch('/api/image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    image: currentImageData,
                    conversation_id: currentConversationId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'processing') {
                    startStatusCheck();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                isProcessing = false;
                loading.classList.add('hidden');
                addMessage(imageConversation, 'An error occurred while processing your image.', 'bot');
            });
        }
    }
    
    function resetConversation(type) {
        if (type === 'text') {
            textConversation.innerHTML = '';
        } else {
            imageConversation.innerHTML = '';
            imagePreview.innerHTML = '';
            currentImageData = null;
            imageSend.disabled = true;
        }
        
        currentConversationId = Date.now().toString();
        
        fetch('/api/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type
            })
        });
    }
    
    function addMessage(container, message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Convert markdown-like formatting
        let formattedMessage = message
            .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
        
        messageDiv.innerHTML = formattedMessage;
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }
    
    function addImageToConversation(container, imageData) {
        const imageDiv = document.createElement('div');
        imageDiv.className = 'image-container';
        imageDiv.innerHTML = `<img src="${imageData}" alt="Uploaded Image">`;
        container.appendChild(imageDiv);
        container.scrollTop = container.scrollHeight;
    }
    
    function startStatusCheck() {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        statusCheckInterval = setInterval(checkResponseStatus, 1000);
    }
    
    function checkResponseStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ready') {
                    clearInterval(statusCheckInterval);
                    isProcessing = false;
                    loading.classList.add('hidden');
                    
                    if (data.error) {
                        const errorMessage = `Error: ${data.error}`;
                        if (textInterface.classList.contains('active')) {
                            addMessage(textConversation, errorMessage, 'bot');
                        } else {
                            addMessage(imageConversation, errorMessage, 'bot');
                        }
                    } else if (data.response) {
                        if (textInterface.classList.contains('active')) {
                            addMessage(textConversation, data.response, 'bot');
                        } else {
                            addMessage(imageConversation, data.response, 'bot');
                        }
                    }
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
            });
    }
});""")

def main():
    """Main function to run the web application."""
    # Create template and static files
    create_templates_and_static()
    
    # Initialize Gemma clients in a separate thread
    threading.Thread(target=initialize_clients).start()
    
    # Create images directory for storing uploaded images
    os.makedirs('images', exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
