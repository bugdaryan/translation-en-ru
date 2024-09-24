import yaml
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset
import wandb
import json


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
    input_text = f"[BOS] {text} [EOS]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=10, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


tokenizer = GPT2TokenizerFast.from_pretrained("./translation_model")
model = GPT2LMHeadModel.from_pretrained("./translation_model", device_map="auto")

input_text = "It turned out that the very opposite was the case."
translated_text = translate(input_text, model, tokenizer, max_length)
print(f"Translated text: {translated_text}")