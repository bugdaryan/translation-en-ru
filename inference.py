import torch
from params import en_token, ru_token, config_model, device
from utils import load_model_tokenizer, postprocess_text, load_ner_model
from flair.data import Sentence

def inference(input_text, translation_mode='normal'):
    model, tokenizer = load_model_tokenizer(config_model['model_name'], en_token, ru_token, device)
    
    if translation_mode == 'undertranslation':
        input_text = undertranslate(input_text)
    
    translated_text = translate(input_text, model, tokenizer, config_model['max_length'])
    
    if translation_mode == 'overtranslation':
        translated_text = overtranslate(translated_text, model, tokenizer, config_model['max_length'])

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


def undertranslate(original_text):
    ner_model = load_ner_model()
    special_words = {}
    for word in original_text.split():
        sentence = Sentence(word.capitalize())
        ner_model.predict(sentence)
        for entity in sentence.get_spans('ner'):
            special_words[word] = entity.text
    for old, new in special_words.items():
        original_text = original_text.replace(old, new)

    return original_text
        


def overtranslate(translated_text, model, tokenizer, max_length):
    ner_model = load_ner_model()
    sentence = Sentence(translated_text)
    ner_model.predict(sentence)
    overtranslation_map = {}
    for entity in sentence.get_spans('ner'):
        overtranslation_map[entity.text] = translate(entity.text, model, tokenizer, max_length)

    for original_text, overtranslated_text in overtranslation_map.items():
        translated_text = translated_text.replace(original_text, overtranslated_text)

    return translated_text
