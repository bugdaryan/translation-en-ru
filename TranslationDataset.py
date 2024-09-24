from params import en_token, ru_token
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, source_lang='en', target_lang='ru'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_text = item['translation'][self.source_lang]
        target_text = item['translation'][self.target_lang]
        
        input_text = f"{en_token} {source_text} {ru_token} {target_text}"
        encodings = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding="max_length")
        encodings["labels"] = encodings["input_ids"].copy()
        
        source_tokens = self.tokenizer(f"{en_token} {source_text} {ru_token}", truncation=True, max_length=self.max_length)
        encodings["labels"][:len(source_tokens["input_ids"])] = [-100] * len(source_tokens["input_ids"])

        return encodings