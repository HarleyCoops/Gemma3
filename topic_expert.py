"""
Gemma 3 Topic Expert

This script demonstrates how to use Gemma 3 to create a specialized chatbot
that can answer questions about a specific topic with enhanced knowledge.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Google GenAI SDK with your API key
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Error: Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    print("You can get an API key from Google AI Studio: https://makersuite.google.com/")
    sys.exit(1)

# Uncomment the following for local JAX execution (if needed)
"""
import gemma as gm
import jax.numpy as jnp

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
"""

class TopicExpert:
    """A specialized chatbot that's an expert on a specific topic."""
    
    def __init__(self, topic, context=None, model_name="gemma-3-8b-it"):
        """
        Initialize the topic expert.
        
        Args:
            topic: The main topic this expert specializes in.
            context: Optional additional context or knowledge about the topic.
            model_name: The Gemma model to use.
        """
        self.topic = topic
        self.context = context
        self.model_name = model_name
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name=model_name)
        
        # Create system prompt
        system_prompt = f"You are an expert on {topic}."
        if context:
            system_prompt += f" Here's some specialized knowledge you have: {context}"
        
        # Initialize chat session with system prompt
        self.chat = self.model.start_chat(history=[])
        self._send_system_prompt(system_prompt)
    
    def _send_system_prompt(self, system_prompt):
        """Send a system prompt to set up the expert's role."""
        try:
            self.chat.send_message(f"I want you to act as an expert on {self.topic}. " + 
                                  f"Please respond to all questions as if you are a leading authority on this subject. " +
                                  f"Keep your answers focused on {self.topic}.")
        except Exception as e:
            print(f"Error setting up expert role: {str(e)}")
    
    def ask(self, question):
        """
        Ask the expert a question about its topic.
        
        Args:
            question: The question to ask.
            
        Returns:
            The expert's response.
        """
        try:
            response = self.chat.send_message(question)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def reset(self):
        """Reset the conversation while maintaining the expert role."""
        self.chat = self.model.start_chat(history=[])
        system_prompt = f"You are an expert on {self.topic}."
        if self.context:
            system_prompt += f" Here's some specialized knowledge you have: {self.context}"
        self._send_system_prompt(system_prompt)


def create_expert_from_file(topic, knowledge_file):
    """
    Create an expert with knowledge loaded from a file.
    
    Args:
        topic: The main topic this expert specializes in.
        knowledge_file: Path to a text file containing specialized knowledge.
        
    Returns:
        A TopicExpert instance.
    """
    try:
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            context = f.read()
        return TopicExpert(topic, context)
    except Exception as e:
        print(f"Error loading knowledge file: {str(e)}")
        return TopicExpert(topic)


def interactive_mode(expert):
    """
    Run the expert in interactive mode.
    
    Args:
        expert: A TopicExpert instance.
    """
    print(f"\n=== Gemma 3 {expert.topic} Expert ===")
    print("Ask questions about this topic or type 'exit' to quit, 'reset' to restart the conversation.")
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if question.lower() == 'reset':
            expert.reset()
            print("Conversation reset.")
            continue
        
        if not question:
            continue
        
        response = expert.ask(question)
        print(f"\nExpert: {response}")


def main():
    """Main function to run the topic expert."""
    parser = argparse.ArgumentParser(description="Gemma 3 Topic Expert")
    parser.add_argument("--topic", type=str, default="artificial intelligence", 
                        help="The topic the expert specializes in")
    parser.add_argument("--knowledge", type=str, 
                        help="Path to a text file containing specialized knowledge")
    parser.add_argument("--model", type=str, default="gemma-3-8b-it",
                        help="Model name to use (default: gemma-3-8b-it)")
    
    args = parser.parse_args()
    
    # Create the expert
    if args.knowledge:
        expert = create_expert_from_file(args.topic, args.knowledge)
    else:
        expert = TopicExpert(args.topic, model_name=args.model)
    
    # Run in interactive mode
    interactive_mode(expert)


if __name__ == "__main__":
    main()
