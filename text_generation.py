"""
Gemma 3 Text Generation Demo

This script demonstrates various text generation capabilities of Gemma 3,
including basic prompting, multi-turn conversations, and different sampling methods.
"""

import os
import gemma as gm
import jax.numpy as jnp

# Set GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

def load_model():
    """Load the Gemma 3 4B instruction-tuned model."""
    print("Loading Gemma 3 4B instruction-tuned model...")
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
    tokenizer = gm.Tokenizer()
    return model, params, tokenizer

def chat_example(model, params, tokenizer):
    """Demonstrate multi-turn conversation using ChatSampler."""
    print("\n=== Multi-turn Conversation Example ===")
    
    chat_sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        multi_turn=True  # Maintain conversation context
    )
    
    # First turn
    prompt = "Explain quantum computing in simple terms."
    print(f"\nUser: {prompt}")
    response = chat_sampler.send_message(prompt)
    print(f"Gemma: {response}")
    
    # Second turn (maintains conversation context)
    prompt = "What are some practical applications?"
    print(f"\nUser: {prompt}")
    response = chat_sampler.send_message(prompt)
    print(f"Gemma: {response}")
    
    # Third turn
    prompt = "How close are we to having practical quantum computers?"
    print(f"\nUser: {prompt}")
    response = chat_sampler.send_message(prompt)
    print(f"Gemma: {response}")
    
    # Reset conversation
    chat_sampler.reset()
    print("\nConversation reset.")

def sampler_example(model, params, tokenizer):
    """Demonstrate using the lower-level Sampler API."""
    print("\n=== Lower-level Sampler API Example ===")
    
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer
    )
    
    # Format prompt with special tokens
    prompt = "\u003cstart_of_turn\u003euser\nWrite a short poem about artificial intelligence.\u003cend_of_turn\u003e\n\u003cstart_of_turn\u003emodel\n"
    print("\nPrompt: Write a short poem about artificial intelligence.")
    
    # Greedy sampling (default)
    print("\n--- Greedy Sampling ---")
    response = sampler.sample(prompt)
    print(response)
    
    # Random sampling with different temperatures
    print("\n--- Random Sampling (Temperature = 0.5) ---")
    response = sampler.sample(
        prompt, 
        sampler=gm.text.RandomSampling(temperature=0.5)
    )
    print(response)
    
    print("\n--- Random Sampling (Temperature = 1.0) ---")
    response = sampler.sample(
        prompt, 
        sampler=gm.text.RandomSampling(temperature=1.0)
    )
    print(response)

def direct_model_example(model, params, tokenizer):
    """Demonstrate direct model usage for token prediction."""
    print("\n=== Direct Model Usage Example ===")
    
    # Encode prompt
    prompt = "One word to describe the future of AI: "
    encoded_prompt = tokenizer.encode(prompt, add_bos=True)
    
    print(f"\nPrompt: {prompt}")
    
    # Generate a single token prediction
    logits = model.apply(params, encoded_prompt)
    next_token = jnp.argmax(logits[:, -1], axis=-1)
    next_token_str = tokenizer.decode([next_token])
    
    print(f"Predicted token: {next_token_str}")
    
    # Generate a few more tokens
    current_tokens = encoded_prompt
    generated_text = prompt
    
    print("\nGenerating a few more tokens...")
    for _ in range(10):
        logits = model.apply(params, current_tokens)
        next_token = jnp.argmax(logits[:, -1], axis=-1)
        next_token_str = tokenizer.decode([next_token])
        generated_text += next_token_str
        current_tokens = jnp.append(current_tokens, next_token)
    
    print(f"Generated text: {generated_text}")

def main():
    """Run all text generation examples."""
    model, params, tokenizer = load_model()
    
    # Run examples
    chat_example(model, params, tokenizer)
    sampler_example(model, params, tokenizer)
    direct_model_example(model, params, tokenizer)
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main()
