import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import wandb
from transformers import EarlyStoppingCallback
import evaluate


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)



batch_size = int(config['training']['batch_size'])
epochs = int(config['training']['epochs'])
learning_rate = float(config['training']['learning_rate'])
warmup_steps = int(config['training']['warmup_steps'])
gradient_accumulation_steps = int(config['training']['gradient_accumulation_steps'])
max_grad_norm = float(config['training']['max_grad_norm'])
max_length = int(config['model']['max_length'])
model_name = config['model']['name'].split("/")[-1] if "/" in config['model']['name'] else config['model']['name']

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
# split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset['train'], dataset['validation']
train_ds = train_ds.shuffle(seed=42)


tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

en_token = "[EN]"
ru_token = "[RU]"
tokenizer.add_special_tokens({'additional_special_tokens': [en_token, ru_token], 'pad_token': '[PAD]'})
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
        
        input_text = f"{en_token} {source_text} {ru_token} {target_text}"
        encodings = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding="max_length")
        
        encodings["labels"] = encodings["input_ids"].copy()
        
        source_tokens = self.tokenizer(f"[EN] {source_text} [RU]", truncation=True, max_length=self.max_length)
        encodings["labels"][:len(source_tokens["input_ids"])] = [-100] * len(source_tokens["input_ids"])

        return encodings

train_dataset = TranslationDataset(train_ds, tokenizer, max_length)
val_dataset = TranslationDataset(val_ds, tokenizer, max_length)

bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_labels = [[label] for label in decoded_labels]
    
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu["bleu"]}

training_args = TrainingArguments(
    output_dir=f"./results/{model_name}",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir=f'./logs/{model_name}',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    # eval_accumulation_steps=50,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    report_to='wandb',
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False,
    deepspeed="ds_config.json",
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping_callback],
    # compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained(f"./results/{model_name}")
tokenizer.save_pretrained(f"./results/{model_name}")