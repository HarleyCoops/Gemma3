"""
Gemma 3 API Integration

This script demonstrates how to use Gemma 3 models through the Google GenAI SDK,
which is more efficient than running the models locally.
"""

import os
import time
from typing import List, Optional
import base64
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Google GenAI SDK with your API key
genai.configure(api_key=API_KEY)

# Uncomment the following for local JAX execution (if needed)
"""
import gemma as gm
import jax.numpy as jnp

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
"""

class GemmaAPI:
    """Class for interacting with Gemma 3 models via Google GenAI SDK."""
    
    def __init__(self, model_name="gemma-3-8b-it"):
        """
        Initialize the Gemma API client.
        
        Args:
            model_name: The model to use. Options include:
                - "gemma-3-8b-it" (instruction-tuned)
                - "gemma-3-8b" (base model)
                - "gemma-3-27b-it" (larger instruction-tuned)
                - "gemma-3-vision-it" (multimodal model)
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name=model_name)
        self.chat_session = self.model.start_chat(history=[])
        
        # For local execution (commented out)
        """
        if "vision" in model_name:
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
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using Gemma 3.
        
        Args:
            prompt: The text prompt to send to the model.
            
        Returns:
            The generated text response.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating text: {str(e)}"
        
        # For local execution (commented out)
        """
        try:
            response = self.chat_sampler.send_message(prompt)
            return response
        except Exception as e:
            return f"Error generating text locally: {str(e)}"
        """
    
    def chat(self, message: str) -> str:
        """
        Send a message in an ongoing chat conversation.
        
        Args:
            message: The message to send.
            
        Returns:
            The model's response.
        """
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def reset_chat(self):
        """Reset the chat conversation history."""
        self.chat_session = self.model.start_chat(history=[])
        
        # For local execution (commented out)
        """
        self.chat_sampler.reset()
        """
    
    def process_image(self, image_path: str, prompt: str) -> str:
        """
        Process an image with a text prompt.
        
        Args:
            image_path: Path to the image file.
            prompt: Text prompt to accompany the image.
            
        Returns:
            The model's response.
        """
        if "vision" not in self.model_name:
            return "Error: You need to use a vision model (e.g., 'gemma-3-vision-it') for image processing."
        
        try:
            # Load and prepare the image
            image = Image.open(image_path)
            
            # Send the multimodal prompt to the model
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error processing image: {str(e)}"
        
        # For local execution (commented out)
        """
        try:
            image = Image.open(image_path)
            response = self.chat_sampler.send_message(prompt, images=[image])
            return response
        except Exception as e:
            return f"Error processing image locally: {str(e)}"
        """
    
    def process_image_base64(self, image_base64: str, prompt: str) -> str:
        """
        Process an image from base64 string with a text prompt.
        
        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt to accompany the image.
            
        Returns:
            The model's response.
        """
        if "vision" not in self.model_name:
            return "Error: You need to use a vision model (e.g., 'gemma-3-vision-it') for image processing."
        
        try:
            # Decode the base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Send the multimodal prompt to the model
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error processing base64 image: {str(e)}"


def text_generation_demo():
    """Demonstrate text generation capabilities."""
    print("\n=== Text Generation Demo ===")
    
    gemma = GemmaAPI(model_name="gemma-3-8b-it")
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the ethical implications of advanced AI systems?",
        "Provide a recipe for a vegetarian lasagna."
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = gemma.generate_text(prompt)
        print(f"Response: {response}")
        print("-" * 80)
        time.sleep(1)  # Avoid rate limiting


def chat_demo():
    """Demonstrate multi-turn chat capabilities."""
    print("\n=== Multi-turn Chat Demo ===")
    
    gemma = GemmaAPI(model_name="gemma-3-8b-it")
    
    conversation = [
        "Tell me about the solar system.",
        "Which planet has the most moons?",
        "What makes Jupiter's atmosphere unique?",
        "How does Jupiter compare to Saturn?"
    ]
    
    for message in conversation:
        print(f"\nUser: {message}")
        response = gemma.chat(message)
        print(f"Gemma: {response}")
        print("-" * 80)
        time.sleep(1)  # Avoid rate limiting
    
    # Reset chat
    print("\nResetting chat...")
    gemma.reset_chat()
    
    # New conversation
    print("\nStarting new conversation...")
    response = gemma.chat("What are some good books to read?")
    print(f"Gemma: {response}")


def multimodal_demo(image_folder="./images"):
    """Demonstrate multimodal capabilities."""
    print("\n=== Multimodal Demo ===")
    
    # Ensure the image folder exists
    os.makedirs(image_folder, exist_ok=True)
    
    # Check if there are images in the folder
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in {image_folder}. Please add some images and run again.")
        return
    
    gemma = GemmaAPI(model_name="gemma-3-vision-it")
    
    for image_file in images[:3]:  # Process up to 3 images
        image_path = os.path.join(image_folder, image_file)
        print(f"\nProcessing image: {image_file}")
        
        # Image captioning
        response = gemma.process_image(image_path, "Describe this image in detail.")
        print(f"Image description: {response}")
        
        # Visual question answering
        response = gemma.process_image(image_path, "What objects can you identify in this image?")
        print(f"Objects identified: {response}")
        
        # Emotional analysis
        response = gemma.process_image(image_path, "What emotions might this image evoke?")
        print(f"Emotional analysis: {response}")
        
        print("-" * 80)
        time.sleep(1)  # Avoid rate limiting


def main():
    """Run all demos."""
    print("Gemma 3 API Integration Demo")
    print("============================")
    
    # Check if API key is available
    if not API_KEY:
        print("Error: Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        print("You can get an API key from Google AI Studio: https://makersuite.google.com/")
        return
    
    # Run demos
    text_generation_demo()
    chat_demo()
    
    # Only run multimodal demo if images are available
    image_folder = "./images"
    if os.path.exists(image_folder) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(image_folder)):
        multimodal_demo(image_folder)
    else:
        print("\n=== Multimodal Demo ===")
        print("Skipping multimodal demo. To run it, create an 'images' folder and add some images.")
    
    print("\nAll demos completed.")


if __name__ == "__main__":
    main()
