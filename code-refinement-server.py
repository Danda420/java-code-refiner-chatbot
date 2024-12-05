from typing import Optional
import re
import os
import torch
import random
import numpy as np
import bitsandbytes
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ngrok.set_auth_token("YOUR NGROK TOKEN")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_java_code(code):
    indent_level = 0
    formatted_code = ""
    for line in re.split(r'(?<=;|{|})\s*', code):
        stripped_line = line.strip()
        if stripped_line.startswith("}"):
            indent_level -= 1
        formatted_code += "    " * indent_level + stripped_line + "\n"
        if stripped_line.endswith("{"):
            indent_level += 1
    return formatted_code

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("Danda245/CodeT5-base_code-refinement")
model = AutoModelForSeq2SeqLM.from_pretrained("Danda245/CodeT5-base_code-refinement").to(device)

# Enable CORS for Flask app
app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "*"}})

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input', '')

    set_seed(41)

    if user_input:
        inputs = tokenizer(
            user_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
        ).to(device)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0) 
            attention_mask = attention_mask.unsqueeze(0) 

        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            temperature=0.7,
            early_stopping=True,
        )
        refined_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        formatted_code = format_java_code(refined_code)

        return jsonify({'answer': formatted_code})


if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")
    app.run(host='0.0.0.0', port=5000)
