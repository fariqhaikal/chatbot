# ui.py

import gradio as gr
from model_loader import load_model
from memory import ChatMemory
from chat import generate_response

tokenizer, model = load_model()
memory = ChatMemory()

def chat_fn(user_input):
    prompt = memory.get_prompt(user_input)
    reply = generate_response(prompt, tokenizer, model)
    memory.add_turn(user_input, reply)
    return reply

iface = gr.Interface(fn=chat_fn, inputs="text", outputs="text", title="Qwen Chatbot")
iface.launch()