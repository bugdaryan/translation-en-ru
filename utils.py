from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import wandb
import re
from safetensors.torch import load_file
from flair.models import SequenceTagger

def init_wandb(config):
    wandb.init(project='translation-model',
            config=config)


def load_model_tokenizer(model_name, tokenizer_name, device):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
    model.to(device)

    return model, tokenizer


def load_ner_model():
    tagger = SequenceTagger.load("flair/ner-english-fast")
    return tagger


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