"""
List Available Models in Google GenAI API

This script lists all available models in the Google GenAI API.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Google GenAI SDK with your API key
genai.configure(api_key=API_KEY)

def main():
    """List all available models."""
    # Check if API key is available
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please make sure you have a valid API key in your .env file.")
        return
    
    print(f"API Key found: {API_KEY[:5]}...{API_KEY[-5:]}")
    print("Listing available models in Google GenAI API...\n")
    
    try:
        # Get the list of available models (returns a generator)
        models = genai.list_models()
        
        # Print model information
        print("Available models:")
        model_count = 0
        
        for model in models:
            model_count += 1
            print(f"\nModel: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Supported Generation Methods: {model.supported_generation_methods}")
            print("-" * 80)
        
        print(f"\nTotal models found: {model_count}")
    
    except Exception as e:
        print(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    main() 