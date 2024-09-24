import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import json
from safetensors.torch import load_file
import re


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

batch_size = int(config['training']['batch_size'])
epochs = int(config['training']['epochs'])
learning_rate = float(config['training']['learning_rate'])
warmup_steps = int(config['training']['warmup_steps'])
gradient_accumulation_steps = int(config['training']['gradient_accumulation_steps'])
max_grad_norm = float(config['training']['max_grad_norm'])
max_length = int(config['model']['max_length'])



def translate(text, model, tokenizer, max_length):
    model.eval()
    input_text = f"[EN] {text} [RU]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    translation = translated_text.replace(text, "").strip()
    words = re.findall(r'\w+|[^\w\s]', translation, re.UNICODE)
    processed_words = []

    for i in range(len(words)):
        if i > 0 and words[i] == words[i-1]:
            break
        if words[i] not in ['?', '!', '.', ','] and len(words[i]) < 20:
            processed_words.append(words[i])
    return translated_text, ' '.join(processed_words)


en_token = "[EN]"
ru_token = "[RU]"
load_safetensors = False
if load_safetensors:
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'], device_map="auto")
    tokenizer.add_special_tokens({'additional_special_tokens': [en_token, ru_token], 'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    state = load_file("./results/TinyLlama_v1.1/checkpoint-200/model.safetensors")
    model.load_state_dict(state)
else:
    tokenizer = AutoTokenizer.from_pretrained('./results/TinyLlama_v1.1')
    model = AutoModelForCausalLM.from_pretrained('./results/TinyLlama_v1.1', device_map="auto")

input_text = "I am eatting apple while i am working at Apple"
full_translation, translated_text = translate(input_text, model, tokenizer, max_length)
print(f"Original text: {input_text}")
print(f"Translated text: {translated_text}")
print(f"Full translation: {full_translation}")