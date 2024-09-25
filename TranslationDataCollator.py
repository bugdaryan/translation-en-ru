from transformers import DataCollatorWithPadding
import torch
import random

class TranslationDataCollator:
    def __init__(self, tokenizer, batch_size, max_length, source_lang, target_lang):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang  

    def __call__(self, examples):
        source_text = [x['translation'][self.source_lang] for x in examples]
        target_text = [x['translation'][self.target_lang] for x in examples]
        
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
            )
        
        model_inputs['labels'] = target_encoding['input_ids']
        
        return model_inputs


