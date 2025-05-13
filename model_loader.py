# model_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, DEVICE

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model
