"""
Gemma 3 AI Assistant

A comprehensive AI assistant that leverages Gemma 3's text and multimodal capabilities
to provide document analysis, image understanding, and interactive conversations.
"""

import os
import sys
import argparse
import gemma as gm
from PIL import Image
import matplotlib.pyplot as plt
import PyPDF2
import docx
import pytesseract
from io import BytesIO
import requests
import time

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

class GemmaAssistant:
    """AI Assistant powered by Gemma 3 models."""
    
    def __init__(self, use_multimodal=False):
        """Initialize the Gemma Assistant with either text-only or multimodal model."""
        self.use_multimodal = use_multimodal
        self.tokenizer = gm.Tokenizer()
        
        if use_multimodal:
            print("Initializing multimodal Gemma 3 assistant...")
            self.model = gm.nn.Gemma3Vision_4B()
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3VISION_4B_IT)
        else:
            print("Initializing text-only Gemma 3 assistant...")
            self.model = gm.nn.Gemma3_4B()
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
        
        # Initialize chat sampler
        self.chat_sampler = gm.text.ChatSampler(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
            multi_turn=True
        )
        
        print("Assistant initialized and ready!")
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.chat_sampler.reset()
        print("Conversation history has been reset.")
    
    def process_text(self, text, prompt=None):
        """Process text with an optional specific prompt."""
        if prompt is None:
            prompt = "Please analyze the following text and provide insights:"
        
        full_prompt = f"{prompt}\n\n{text}"
        response = self.chat_sampler.send_message(full_prompt)
        return response
    
    def process_image(self, image, prompt=None):
        """Process an image with an optional specific prompt."""
        if not self.use_multimodal:
            return "Error: This assistant is not initialized with multimodal capabilities."
        
        if prompt is None:
            prompt = "Please analyze this image and describe what you see:"
        
        response = self.chat_sampler.send_message(prompt, images=[image])
        return response
    
    def chat(self, message, image=None):
        """Have a conversation with the assistant, optionally including an image."""
        if image and not self.use_multimodal:
            return "Error: This assistant is not initialized with multimodal capabilities."
        
        if image:
            response = self.chat_sampler.send_message(message, images=[image])
        else:
            response = self.chat_sampler.send_message(message)
        
        return response
    
    def summarize_document(self, file_path):
        """Extract and summarize text from various document formats."""
        text = self.extract_text_from_document(file_path)
        if not text:
            return "Error: Could not extract text from the document."
        
        # If text is very long, split into chunks
        if len(text) > 10000:
            chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
            summaries = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                prompt = f"This is part {i+1} of {len(chunks)} of a document. Please summarize this section:"
                summary = self.process_text(chunk, prompt)
                summaries.append(summary)
            
            # Combine summaries
            combined_summaries = "\n\n".join(summaries)
            final_summary = self.process_text(
                combined_summaries, 
                "The following are summaries of different parts of a document. Please provide a coherent overall summary:"
            )
            return final_summary
        else:
            return self.process_text(text, "Please summarize this document:")
    
    def extract_text_from_document(self, file_path):
        """Extract text from various document formats (PDF, DOCX, TXT)."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                return self._extract_from_image(file_path)
            else:
                return f"Unsupported file format: {file_extension}"
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX."""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_from_txt(self, file_path):
        """Extract text from TXT."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_from_image(self, file_path):
        """Extract text from image using OCR."""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    
    def analyze_image_document(self, file_path, extract_text=True):
        """Analyze an image document, optionally extracting and analyzing text."""
        try:
            if not self.use_multimodal:
                return "Error: This assistant is not initialized with multimodal capabilities."
            
            image = Image.open(file_path)
            
            # Visual analysis
            visual_analysis = self.process_image(image, "Analyze this document visually:")
            
            # Text extraction and analysis if requested
            if extract_text:
                extracted_text = self._extract_from_image(file_path)
                if extracted_text.strip():
                    text_analysis = self.process_text(
                        extracted_text, 
                        "The following text was extracted from the document image. Please analyze it:"
                    )
                    
                    # Combined analysis
                    combined_prompt = (
                        "You've analyzed this document both visually and through its text content. "
                        "Please provide a comprehensive analysis combining both perspectives:"
                    )
                    self.chat_sampler.send_message(combined_prompt)
                    return {
                        "visual_analysis": visual_analysis,
                        "extracted_text": extracted_text,
                        "text_analysis": text_analysis,
                        "combined_analysis": self.chat_sampler.send_message(combined_prompt)
                    }
                
            return {
                "visual_analysis": visual_analysis,
                "extracted_text": "No text extracted or text extraction not requested."
            }
            
        except Exception as e:
            return f"Error analyzing image document: {str(e)}"

def interactive_mode(assistant):
    """Run the assistant in interactive mode."""
    print("\n=== Gemma 3 AI Assistant Interactive Mode ===")
    print("Type 'exit' to quit, 'reset' to clear conversation history, or 'image' to process an image.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        
        elif user_input.lower() == 'reset':
            assistant.reset_conversation()
            continue
        
        elif user_input.lower() == 'image':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    prompt = input("Enter prompt (or press Enter for default): ").strip()
                    prompt = prompt if prompt else None
                    
                    print("\nProcessing image...")
                    response = assistant.process_image(image, prompt)
                    print(f"\nGemma: {response}")
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
            else:
                print(f"Image not found: {image_path}")
        
        elif user_input.lower().startswith('document '):
            doc_path = user_input[9:].strip()
            if os.path.exists(doc_path):
                print("\nAnalyzing document...")
                response = assistant.summarize_document(doc_path)
                print(f"\nGemma: {response}")
            else:
                print(f"Document not found: {doc_path}")
        
        else:
            response = assistant.chat(user_input)
            print(f"\nGemma: {response}")

def main():
    """Main function to run the Gemma 3 AI Assistant."""
    parser = argparse.ArgumentParser(description="Gemma 3 AI Assistant")
    parser.add_argument("--multimodal", action="store_true", help="Use multimodal model")
    parser.add_argument("--document", type=str, help="Path to document to analyze")
    parser.add_argument("--image", type=str, help="Path to image to analyze")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = GemmaAssistant(use_multimodal=args.multimodal or args.image is not None)
    
    # Process document if provided
    if args.document:
        if os.path.exists(args.document):
            print(f"\nAnalyzing document: {args.document}")
            result = assistant.summarize_document(args.document)
            print(f"\nDocument Analysis:\n{result}")
        else:
            print(f"Document not found: {args.document}")
    
    # Process image if provided
    if args.image:
        if os.path.exists(args.image):
            print(f"\nAnalyzing image: {args.image}")
            image = Image.open(args.image)
            result = assistant.process_image(image)
            print(f"\nImage Analysis:\n{result}")
        else:
            print(f"Image not found: {args.image}")
    
    # Run in interactive mode if requested or if no specific task
    if args.interactive or (not args.document and not args.image):
        interactive_mode(assistant)

if __name__ == "__main__":
    main()
