# Gemma 3 Guide

## Overview
This guide provides comprehensive documentation for using Google's Gemma 3 models for text generation and multimodal tasks. Gemma 3 offers powerful language modeling capabilities with both text-only and multimodal versions.

## Installation

```bash
pip install -q gemma
```

## Environment Setup

Set GPU memory allocation:
```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
```

Required libraries:
- `gemma`
- `jax`
- `jax.numpy`
- `tensorflow_datasets` (for multimodal examples)

## Text Generation

### Loading a Model

```python
import gemma as gm

# Load the instruction-tuned 4B model
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
```

### Methods for Prompting

Gemma 3 provides three main methods for prompting the model:

#### 1. Using ChatSampler (Recommended for Conversations)

```python
# Initialize the chat sampler
chat_sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    tokenizer=gm.Tokenizer(),
    multi_turn=True  # Set to True to maintain conversation context
)

# First turn
response = chat_sampler.send_message("Tell me about Paris.")
print(response)

# Second turn (maintains conversation context)
response = chat_sampler.send_message("What are some must-visit attractions?")
print(response)

# Reset conversation
chat_sampler.reset()
```

#### 2. Using Sampler (Lower-level API)

```python
# Initialize the sampler
sampler = gm.text.Sampler(
    model=model,
    params=params,
    tokenizer=gm.Tokenizer()
)

# Format prompt with special tokens
prompt = "\u003cstart_of_turn\u003euser\nTell me about Paris.\u003cend_of_turn\u003e\n\u003cstart_of_turn\u003emodel\n"
response = sampler.sample(prompt)
print(response)
```

#### 3. Direct Model Usage

```python
# Initialize tokenizer
tokenizer = gm.Tokenizer()

# Encode prompt
prompt = tokenizer.encode("One word to describe Paris: \n\n", add_bos=True)

# Generate a single token prediction
logits = model.apply(params, prompt)
next_token = jnp.argmax(logits[:, -1], axis=-1)
next_token_str = tokenizer.decode([next_token])
print(next_token_str)
```

### Sampling Methods

#### Greedy Decoding (Default)
```python
# Uses argmax to select the most likely token at each step
response = sampler.sample(prompt)
```

#### Random Sampling
```python
# Introduces variety in responses
response = sampler.sample(
    prompt, 
    sampler=gm.text.RandomSampling(temperature=0.7)
)
```

## Multimodal Capabilities

Gemma 3 supports multimodal inputs combining text and images.

### Loading a Multimodal Model

```python
import gemma as gm
import tensorflow_datasets as tfds
from PIL import Image

# Load the multimodal model
model = gm.nn.Gemma3Vision_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3VISION_4B_IT)
```

### Processing Images

```python
# Load an image
image = Image.open("path/to/image.jpg")

# Initialize multimodal sampler
mm_sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    tokenizer=gm.Tokenizer(),
    multi_turn=True
)

# Send a message with an image
response = mm_sampler.send_message("Describe this image.", images=[image])
print(response)
```

## Best Practices

1. **Memory Management**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION` to control GPU memory allocation.
2. **Conversation Context**: Use `multi_turn=True` with ChatSampler to maintain context across turns.
3. **Response Variety**: Adjust temperature in RandomSampling for more diverse responses.
4. **Prompt Engineering**: Format prompts correctly with special tokens when using lower-level APIs.

## Examples

### Multi-turn Conversation
```python
chat_sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    tokenizer=gm.Tokenizer(),
    multi_turn=True
)

print(chat_sampler.send_message("Tell me a short story about a robot."))
print(chat_sampler.send_message("Continue the story with a plot twist."))
print(chat_sampler.send_message("How does the story end?"))
```

### Image Analysis
```python
mm_sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    tokenizer=gm.Tokenizer(),
    multi_turn=True
)

image = Image.open("sunset.jpg")
print(mm_sampler.send_message("What time of day is shown in this image?", images=[image]))
print(mm_sampler.send_message("What emotions does this scene evoke?"))
```

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or model size, or increase memory allocation.
- **Slow Inference**: Consider using a smaller model or optimizing with XLA compilation.
- **Unexpected Responses**: Check prompt formatting, especially when using lower-level APIs.

## Resources

- [Gemma Official Documentation](https://ai.google.dev/gemma)
- [JAX Documentation](https://jax.readthedocs.io/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)

## Available Scripts

This project includes several Python scripts for different functionalities:

### Core Scripts
- `text_generation.py`: Basic text generation functionality using Gemma 3 models
- `multimodal_vision.py`: Handles multimodal (text + image) processing with Gemma 3 Vision
- `ai_assistant.py`: Implements an AI assistant interface using Gemma 3
- `gemma_api.py`: Provides a RESTful API interface for Gemma 3 functionality

### Web Interfaces
- `web_interface.py`: Basic web interface for interacting with Gemma 3
- `gemma_web_app.py`: Advanced web application with extended features
- `topic_expert.py`: Specialized script for topic-based interactions

### Configuration Files
- `.env`: Environment configuration file (create from sample.env)
- `sample.env`: Template for environment variables
- `requirements.txt`: Core project dependencies
- `requirements_new.txt`: Updated/alternative dependencies

## Running the Scripts

### Text Generation
```bash
python text_generation.py
```

### Web Interface
```bash
python web_interface.py
```

### API Server
```bash
python gemma_api.py
```

### Advanced Web Application
```bash
python gemma_web_app.py
```

### Multimodal Processing
```bash
python multimodal_vision.py
```

### AI Assistant
```bash
python ai_assistant.py
```

### Topic Expert
```bash
python topic_expert.py
```

Make sure to:
1. Install the required dependencies: `pip install -r requirements.txt`
2. Copy `sample.env` to `.env` and configure your environment variables
3. Set up your GPU environment as described in the Environment Setup section
