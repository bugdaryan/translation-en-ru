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
        outputs = model.generate(**inputs, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = translated_text.split("[RU]")[1]
    words = re.findall(r'\w+|[^\w\s]', translation, re.UNICODE)
    processed_words = []

    for i in range(len(words)):
        if i > 0 and words[i] == words[i-1]:
            break
        processed_words.append(words[i])
    return translated_text, ' '.join(processed_words)


tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
model = AutoModelForCausalLM.from_pretrained(config['model']['name'], device_map="auto")

tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
model.resize_token_embeddings(len(tokenizer))

state = load_file("./results/TinyLlama_v1.1/checkpoint-100/model.safetensors")
model.load_state_dict(state)

input_text = "I am working at apple"
full_translation, translated_text = translate(input_text, model, tokenizer, max_length)
print(f"Original text: {input_text}")
print(f"Translated text: {translated_text}")
print(f"Full translation: {full_translation}")