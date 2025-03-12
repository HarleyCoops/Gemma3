"""
Gemma 3 API Test Script

This script tests the connection to the Gemma 3 API and runs a few simple tests.
"""

import os
from dotenv import load_dotenv
from gemma_api import GemmaAPI

# Load environment variables from .env file
load_dotenv()

def test_text_generation():
    """Test basic text generation capabilities."""
    print("\n=== Testing Text Generation ===")
    
    gemma = GemmaAPI(model_name="models/gemma-3-27b-it")
    
    prompt = "Explain what makes Gemma 3 different from other language models in 3 bullet points."
    print(f"\nPrompt: {prompt}")
    
    response = gemma.generate_text(prompt)
    print(f"Response: {response}")

def test_chat():
    """Test multi-turn chat capabilities."""
    print("\n=== Testing Multi-turn Chat ===")
    
    gemma = GemmaAPI(model_name="models/gemma-3-27b-it")
    
    # First message
    message = "What are the key features of Gemma 3?"
    print(f"\nUser: {message}")
    response = gemma.chat(message)
    print(f"Gemma: {response}")
    
    # Follow-up question
    message = "How does the context window size compare to previous models?"
    print(f"\nUser: {message}")
    response = gemma.chat(message)
    print(f"Gemma: {response}")

def main():
    """Run all tests."""
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please make sure you have a valid API key in your .env file.")
        return
    
    print(f"API Key found: {api_key[:5]}...{api_key[-5:]}")
    print("Running Gemma 3 API tests...")
    
    # Run tests
    test_text_generation()
    test_chat()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 