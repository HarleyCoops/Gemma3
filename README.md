# Gemma 3 Guide

## Overview
This project provides two ways to interact with Gemma 3 models:

1. **Local JAX Setup**: Run Gemma 3 models locally using JAX for high-performance machine learning
2. **Google API Integration**: Access Gemma 3 through Google's API service for cloud-based inference

Choose the approach that best fits your needs:
- **Local Setup**: Better for offline use, full control, and custom modifications
- **API Integration**: Easier to get started, no GPU required, and managed by Google

This guide provides comprehensive documentation for using Google's Gemma 3 models for text generation and multimodal tasks. Gemma 3 offers powerful language modeling capabilities with both text-only and multimodal versions.

### Key Features of Gemma 3

- **128K Token Context Window**: 16x larger than previous Gemma models, allowing processing of multiple articles or hundreds of images in a single prompt
- **Multimodal Capabilities**: Handle both image and text input for complex analysis and generation tasks (except 1B model which is text-only)
- **Wide Language Support**: Built-in support for over 140 languages
- **Multiple Model Sizes**: Available in 4 parameter sizes (1B, 4B, 12B, 27B) with 5 precision levels to fit your computational resources

You can download Gemma 3 models from:
- [Kaggle](https://www.kaggle.com/models?query=gemma3&publisher=google)
- [Hugging Face](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)

For technical details, see the [Model Card](https://ai.google.dev/gemma/docs/core/model_card_3) and [Technical Report](https://goo.gle/Gemma3Report).

## Installation

### For Local JAX Setup
```bash
pip install -q gemma
```

### For Google API Integration
```bash
pip install -r requirements.txt
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

Gemma 3 supports multimodal inputs combining text and images. **Note**: The 1B model is text-only and does not support image input.

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

## Model Sizes and Memory Requirements

Gemma 3 models are available in 4 parameter sizes with 5 precision levels:

| **Parameters** | **Full 32bit** | **BF16 (16-bit)** | **SFP8 (8-bit)** | **Q4_0 (4-bit)** | **INT4 (4-bit)** |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 1B (text only) | 4 GB | 1.5 GB | 1.1 GB | 892 MB | 861 MB |
| Gemma 3 4B | 16 GB | 6.4 GB | 4.4 GB | 3.4 GB | 3.2 GB |
| Gemma 3 12B | 48 GB | 20 GB | 12.2 GB | 8.7 GB | 8.2 GB |
| Gemma 3 27B | 108 GB | 46.4 GB | 29.1 GB | 21 GB | 19.9 GB |

Note: These estimates do not include additional memory required for prompt tokens or supporting software. Memory consumption increases based on the total number of tokens in your prompt.

## Best Practices

1. **Memory Management**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION` to control GPU memory allocation.
2. **Conversation Context**: Use `multi_turn=True` with ChatSampler to maintain context across turns.
3. **Response Variety**: Adjust temperature in RandomSampling for more diverse responses.
4. **Prompt Engineering**: Format prompts correctly with special tokens when using lower-level APIs.
5. **Model Selection**: Choose the appropriate model size and precision level based on your hardware capabilities and task requirements.
6. **Context Window**: Leverage the 128K token context window for processing large documents or multiple images.
7. **Language Support**: Utilize the multilingual capabilities with over 140 supported languages.

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
- [Gemma 3 Model Card](https://ai.google.dev/gemma/docs/core/model_card_3)
- [Gemma 3 Technical Report](https://goo.gle/Gemma3Report)
- [JAX Documentation](https://jax.readthedocs.io/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Kaggle Models](https://www.kaggle.com/models?query=gemma3&publisher=google)
- [Hugging Face Models](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)

## Available Scripts

This project includes several Python scripts supporting both local and API-based implementations:

### Google API Scripts
- `gemma_api.py`: RESTful API interface using Google's Gemma API
- `gemma_web_app.py`: Advanced web application with Google API integration
- `web_interface.py`: Basic web interface using Google's API
- `topic_expert.py`: Specialized topic-based interactions via API

### Local JAX Implementation Scripts
- `text_generation.py`: Local text generation using JAX-based Gemma models
- `multimodal_vision.py`: Local multimodal processing with Gemma Vision
- `ai_assistant.py`: Local AI assistant implementation

### Configuration Files
- `.env`: Environment configuration file (create from sample.env)
- `sample.env`: Template for environment variables (includes both API keys and local settings)
- `requirements.txt`: Core project dependencies
- `requirements_new.txt`: Updated/alternative dependencies

## Running the Scripts

### Google API-Based Scripts
These scripts require a Google API key in your `.env` file:
```bash
python gemma_api.py        # Run the API server
python gemma_web_app.py    # Run the advanced web interface
python web_interface.py    # Run the basic web interface
python topic_expert.py     # Run the topic expert system
```

### Local JAX-Based Scripts
These scripts require GPU setup and local model weights:
```bash
python text_generation.py     # Run local text generation
python multimodal_vision.py   # Run local multimodal processing
python ai_assistant.py        # Run local AI assistant
```

## Setup Requirements

### For Google API Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `sample.env` to `.env` and add your Google API key
3. No GPU required

### For Local JAX Setup
1. Install JAX dependencies: `pip install -q gemma`
2. Set up GPU environment as described in the Environment Setup section
3. Configure GPU memory allocation
4. Download model weights locally

Make sure to:
1. Install the required dependencies: `
