import torch
from params import config_model, device
from utils import (
    load_model_tokenizer, 
    load_ner_model, 
    load_embedding_model,
    get_words_embedding_map, 
    get_most_similar_word,
    search_special_words_by_capitalization
)
from flair.data import Sentence

def inference(input_text, translation_mode='normal'):
    model, tokenizer = load_model_tokenizer(config_model['model_name'], config_model['tokenizer_name'], device)
    translated_text = translate(input_text, model, tokenizer, config_model['max_length'])
    
    if translation_mode == 'overtranslation':
        translated_text = overtranslate(translated_text, model, tokenizer, config_model['max_length'])
    elif translation_mode == 'undertranslation':
        translated_text = undertranslate(input_text, translated_text, model, tokenizer, config_model['max_length'])

    return translated_text
    

def translate(text, model, tokenizer, max_length):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


def undertranslate(input_text, translated_text, model, tokenizer, max_length):
    ner_model = load_ner_model()
    special_words = search_special_words_by_capitalization(ner_model, input_text)
    special_words_translated = {}
    for word in special_words:
        special_words_translated[word] = translate(word, model, tokenizer, max_length)

    emb_model, emb_tokenizer = load_embedding_model(config_model['embedding_model_name'], device)

    translated_words = translated_text.split()
    translated_words_embeddings = get_words_embedding_map(emb_model, emb_tokenizer, translated_words)
    special_words_embeddings = get_words_embedding_map(emb_model, emb_tokenizer, special_words_translated.values())

    replace_map = {}
    for special_word in special_words_translated:
        special_word_translated = special_words_translated[special_word]
        replace_map[special_word] = get_most_similar_word(special_words_embeddings[special_word_translated], translated_words_embeddings)

    for special_word in replace_map:
        word_to_replace = replace_map[special_word]
        word_to_replace_with = special_words[special_word]
        translated_text = translated_text.replace(word_to_replace, word_to_replace_with)

    return translated_text
        


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
