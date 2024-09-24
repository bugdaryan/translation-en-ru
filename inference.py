import torch
from params import en_token, ru_token, config_model, device
from utils import load_model_tokenizer, postprocess_text

def inference(input_text):
    model, tokenizer = load_model_tokenizer(config_model['model_name'], en_token, ru_token, device)
    translated_text = translate(input_text, model, tokenizer, config_model['max_length'])
    return translated_text

def translate(text, model, tokenizer, max_length):
    model.eval()
    input_text = f'{en_token} {text} {ru_token}'
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    translation = translated_text.replace(text, '').strip()
    processed_text = postprocess_text(translation)
    return processed_text


# load_safetensors = False
# if load_safetensors:
#     tokenizer = AutoTokenizer.from_pretrained(config_model['name'])
#     model = AutoModelForCausalLM.from_pretrained(config_model['name'], device_map='auto')
#     tokenizer.add_special_tokens({'additional_special_tokens': [en_token, ru_token], 'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))

#     state = load_file('./results/TinyLlama_v1.1/checkpoint-200/model.safetensors')
#     model.load_state_dict(state)
# else:
#     tokenizer = AutoTokenizer.from_pretrained('./results/TinyLlama_v1.1')
#     model = AutoModelForCausalLM.from_pretrained('./results/TinyLlama_v1.1', device_map='auto')

# input_text = 'I am eatting apple while i am working at Apple'
# full_translation, translated_text = translate(input_text, model, tokenizer, max_length)
# print(f'Original text: {input_text}')
# print(f'Translated text: {translated_text}')
# print(f'Full translation: {full_translation}')