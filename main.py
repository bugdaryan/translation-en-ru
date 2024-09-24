import yaml
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
import wandb

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)



batch_size = int(config['training']['batch_size'])
epochs = int(config['training']['epochs'])
learning_rate = float(config['training']['learning_rate'])
warmup_steps = int(config['training']['warmup_steps'])
gradient_accumulation_steps = int(config['training']['gradient_accumulation_steps'])
max_grad_norm = float(config['training']['max_grad_norm'])
max_length = int(config['model']['max_length'])

wandb.init(project="translation-model",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "max_length": max_length,
        })

dataset = load_dataset(config['dataset']['name'], config['dataset']['subset'])
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split_dataset['train'], split_dataset['test']
train_ds = train_ds.shuffle(seed=42)


tokenizer = GPT2TokenizerFast.from_pretrained(config['model']['name'])
model = GPT2LMHeadModel.from_pretrained(config['model']['name'])

tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
model.resize_token_embeddings(len(tokenizer))

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_text = item['translation'][config['dataset']['source_lang']]
        target_text = item['translation'][config['dataset']['target_lang']]
        
        # Tokenize the input
        input_text = f"[BOS] {source_text} [EOS] {target_text} [EOS]"
        encodings = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding="max_length")
        
        encodings["labels"] = encodings["input_ids"].copy()
        
        source_tokens = self.tokenizer(f"[BOS] {source_text} [EOS]", truncation=True, max_length=self.max_length)
        encodings["labels"][:len(source_tokens["input_ids"])] = [-100] * len(source_tokens["input_ids"])

        return encodings

train_dataset = TranslationDataset(train_ds, tokenizer, max_length)
val_dataset = TranslationDataset(val_ds, tokenizer, max_length)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=torch.cuda.is_available(),
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    report_to='wandb'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
model.save_pretrained("./translation_model")
tokenizer.save_pretrained("./translation_model")