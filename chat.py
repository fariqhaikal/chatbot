# chat.py

from config import MAX_TOKENS, DEVICE

def generate_response(prompt, tokenizer, model):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate the response, ensuring to stop after the first sentence
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # Ensure stopping at end-of-sequence token
    )

    # Decode the generated tokens and get only the first response
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return generated.strip().split("\n")[0]  # Return only the first line
