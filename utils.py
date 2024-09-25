from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import wandb
import re
from flair.models import SequenceTagger
import torch
import torch.nn.functional as F
from flair.data import Sentence
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


def load_embedding_model(model_name, device):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_contextual_embeddings(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors='pt')
    tokens = {k: v.to('cuda') for k, v in tokens.items()}
    with torch.no_grad():
        embeddings_out = model(**tokens)
        embeddings = average_pool(embeddings_out.last_hidden_state, tokens['attention_mask'])
    return embeddings


def get_words_embedding_map(model, tokenizer, words):
    embeddings_map = {}
    for word in words:
        embeddings = get_contextual_embeddings(model, tokenizer, word)
        embeddings_map[word] = embeddings
    return embeddings_map


def get_most_similar_word(source_embedding, target_embeddings_map):
    max_similarity = 0
    most_similar_word = None
    for word in target_embeddings_map:
        similarity = F.cosine_similarity(source_embedding, target_embeddings_map[word], dim=1)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_word = word
    return most_similar_word


def search_special_words_by_capitalization(ner_model,text):
    special_words = {}
    for word in text.split():
        sentence = Sentence(word.capitalize())
        ner_model.predict(sentence)
        for entity in sentence.get_spans('ner'):
            special_words[word] = entity.text

    return special_words