# config.py
import torch

# Use GPT-2 (free model) from Hugging Face
MODEL_NAME = "gpt2"  # You can switch to other models like "EleutherAI/gpt-neo-2.7B" if needed
MAX_TOKENS = 100  # Max tokens for generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU
