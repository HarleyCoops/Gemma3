"""
Gemma 3 Multimodal Vision Demo

This script demonstrates the multimodal capabilities of Gemma 3,
including image understanding, captioning, and visual question answering.
"""

import os
import gemma as gm
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

def load_multimodal_model():
    """Load the Gemma 3 Vision 4B instruction-tuned model."""
    print("Loading Gemma 3 Vision 4B instruction-tuned model...")
    model = gm.nn.Gemma3Vision_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3VISION_4B_IT)
    tokenizer = gm.Tokenizer()
    return model, params, tokenizer

def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def display_image(image):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def image_captioning(model, params, tokenizer, image):
    """Generate a caption for an image."""
    print("\n=== Image Captioning Example ===")
    
    mm_sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        multi_turn=True
    )
    
    prompt = "Describe this image in detail."
    print(f"\nPrompt: {prompt}")
    
    response = mm_sampler.send_message(prompt, images=[image])
    print(f"Gemma: {response}")
    
    return mm_sampler  # Return sampler for continued conversation

def visual_qa(mm_sampler, image, question):
    """Answer questions about an image."""
    print(f"\nUser: {question}")
    response = mm_sampler.send_message(question)
    print(f"Gemma: {response}")

def image_analysis(model, params, tokenizer):
    """Perform various image analysis tasks."""
    print("\n=== Image Analysis Examples ===")
    
    # Example 1: Nature scene
    print("\nExample 1: Nature scene")
    nature_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
    try:
        nature_image = download_image(nature_url)
        display_image(nature_image)
        
        mm_sampler = image_captioning(model, params, tokenizer, nature_image)
        
        # Follow-up questions about the image
        visual_qa(mm_sampler, nature_image, "What time of day does this appear to be?")
        visual_qa(mm_sampler, nature_image, "What emotions might someone feel looking at this landscape?")
        visual_qa(mm_sampler, nature_image, "What natural elements are visible in this image?")
        
        # Reset conversation
        mm_sampler.reset()
        
    except Exception as e:
        print(f"Error processing nature image: {e}")
    
    # Example 2: Urban scene
    print("\nExample 2: Urban scene")
    urban_url = "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b"
    try:
        urban_image = download_image(urban_url)
        display_image(urban_image)
        
        mm_sampler = image_captioning(model, params, tokenizer, urban_image)
        
        # Follow-up questions about the image
        visual_qa(mm_sampler, urban_image, "What city might this be?")
        visual_qa(mm_sampler, urban_image, "What architectural styles are visible?")
        visual_qa(mm_sampler, urban_image, "Compare this to a typical rural setting.")
        
        # Reset conversation
        mm_sampler.reset()
        
    except Exception as e:
        print(f"Error processing urban image: {e}")

def multi_image_comparison(model, params, tokenizer):
    """Compare multiple images in the same conversation."""
    print("\n=== Multi-Image Comparison Example ===")
    
    try:
        # Download two different images
        image1_url = "https://images.unsplash.com/photo-1501854140801-50d01698950b"  # Beach
        image2_url = "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b"  # Mountain
        
        image1 = download_image(image1_url)
        image2 = download_image(image2_url)
        
        print("\nImage 1:")
        display_image(image1)
        
        print("\nImage 2:")
        display_image(image2)
        
        # Initialize multimodal sampler
        mm_sampler = gm.text.ChatSampler(
            model=model,
            params=params,
            tokenizer=tokenizer,
            multi_turn=True
        )
        
        # First image
        print("\nAnalyzing first image...")
        response = mm_sampler.send_message("Describe this beach scene.", images=[image1])
        print(f"Gemma: {response}")
        
        # Second image
        print("\nAnalyzing second image...")
        response = mm_sampler.send_message("Now describe this mountain scene.", images=[image2])
        print(f"Gemma: {response}")
        
        # Compare both
        print("\nComparing both images...")
        response = mm_sampler.send_message("Compare and contrast these two landscapes. What are the key differences?")
        print(f"Gemma: {response}")
        
    except Exception as e:
        print(f"Error in multi-image comparison: {e}")

def main():
    """Run all multimodal examples."""
    model, params, tokenizer = load_multimodal_model()
    
    # Run examples
    image_analysis(model, params, tokenizer)
    multi_image_comparison(model, params, tokenizer)
    
    print("\nAll multimodal examples completed.")

if __name__ == "__main__":
    main()
