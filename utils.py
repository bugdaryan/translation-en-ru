from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import re
from safetensors.torch import load_file

def init_wandb(config):
    wandb.init(project='translation-model',
            config=config)


def load_model_tokenizer(model_name, en_token, ru_token, device, checkpoint=None):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = tokenizer.special_tokens_map['additional_special_tokens']
    
    if en_token not in special_tokens \
        and ru_token not in special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [en_token, ru_token], 'pad_token': '[PAD]'})
    
    if len(tokenizer) != len(model.get_input_embeddings().weight):
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if checkpoint:
        model.load_state_dict(load_file(checkpoint))

    return model, tokenizer


def load_train_val_datasets(dataset_name, subset, val_rows=200):
    dataset = load_dataset(dataset_name, subset)
    train_ds, val_ds = dataset['train'], dataset['validation'].select(range(val_rows))
    train_ds = train_ds.shuffle(seed=42)

    return train_ds, val_ds


def postprocess_text(text):
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    processed_words = []

    for i in range(len(words)):
        if i > 0 and words[i] == words[i-1]:
            break
        if words[i] not in ['?', '!', '.', ','] and len(words[i]) < 20:
            processed_words.append(words[i])
    return ' '.join(processed_words)