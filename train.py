from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from TranslationDataset import TranslationDataset
from utils import load_train_val_datasets, init_wandb, load_model_tokenizer
from metrics import compute_metrics
from params import (
    en_token,
    ru_token,
    metric_greater_is_better,
    config_dataset, 
    config_model, 
    config_train, 
    config
)


def train():
    init_wandb(config)
    train_ds, val_ds = load_train_val_datasets(config_dataset['name'], config_dataset['subset'])

    model, tokenizer = load_model_tokenizer(config_model['name'], en_token, ru_token)
    train_dataset = TranslationDataset(train_ds, tokenizer, config_model['max_length'])
    val_dataset = TranslationDataset(val_ds, tokenizer, config_model['max_length'])

    model_name = config_model['name'].split("/")[-1] if "/" in config_model['name'] else config_model['name']

    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        num_train_epochs=config_train['epochs'],
        per_device_train_batch_size=config_train['batch_size'],
        per_device_eval_batch_size=config_train['eval_batch_size'],
        warmup_steps=config_train['warmup_steps'],
        weight_decay=config_train['weight_decay'],
        logging_dir=f'./logs/{model_name}',
        logging_steps=config_train['logging_steps'],
        eval_strategy="steps",
        eval_steps=config_train['eval_steps'],
        save_steps=config_train['save_steps'],
        eval_accumulation_steps=config_train['eval_accumulation_steps'],
        gradient_accumulation_steps=config_train['gradient_accumulation_steps'],
        bf16=config_train['bf16'],
        learning_rate=config_train['learning_rate'],
        max_grad_norm=config_train['max_grad_norm'],
        report_to='wandb',
        load_best_model_at_end=True,
        metric_for_best_model=config_train['metric_for_best_model'],
        greater_is_better=metric_greater_is_better,
        deepspeed=config_train['deepspeed'],
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config_train['early_stopping_patience'],
        early_stopping_threshold=config_train['early_stopping_threshold']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()
    model.save_pretrained(f"./results/{model_name}")
    tokenizer.save_pretrained(f"./results/{model_name}")