# English to Russian Translation Model

This repository contains a simple model for English-to-Russian translation. It provides functionality for both training the model and performing inference with options for normal translation, undertranslation, and overtranslation.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/translation-en-ru.git
   cd translation-en-ru
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, use the following command:

```
python main.py --mode train [OPTIONS]
```

Options:
- `--model_name`: Name of the model to use (default: Helsinki-NLP/opus-mt-en-ru)
- `--dataset_name`: Name of the dataset to use (default: Helsinki-NLP/opus-100)
- `--dataset_subset`: Subset of the dataset (default: en-ru)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate for training (default: 5e-5)
- `--training_batch_size`: Batch size for training (default: 64)
- `--eval_batch_size`: Batch size for evaluation (default: 4)
- `--metric_for_best_model`: Metric to use for selecting the best model (default: meteor)
- `--early_stopping_patience`: Number of evaluation calls with no improvement after which training will be stopped (default: 30)

Example:

```
python main.py --mode train --model_name Helsinki-NLP/opus-mt-en-ru --metric_for_best_model meteor --early_stopping_patience 30
```

### Inference

To perform inference (translation), use the following command:

```
python main.py --mode inference --input_text "Your text here" [OPTIONS]
```

Options:
- `--model_name`: Name of the model to use (default: Helsinki-NLP/opus-mt-en-ru)
- `--translation_mode`: Mode of translation (choices: normal, undertranslation, overtranslation; default: normal)

Example:
```
python main.py --mode inference --model_name Helsinki-NLP/opus-mt-en-ru --input_text "I got a job at Google" --translation_mode overtranslation
```

## Configuration

The main configuration file is `config.yml`. You can modify this file to change various parameters of the model, dataset, and training process. Here are some key sections:

1. Dataset configuration:
   - `name`: Name of the dataset
   - `subset`: Subset of the dataset
   - `source_lang`: Source language code
   - `target_lang`: Target language code

2. Model configuration:
   - `model_name`: Name of the translation model
   - `tokenizer_name`: Name of the tokenizer
   - `max_length`: Maximum sequence length

3. Training configuration:
   - `training_batch_size`: Batch size for training
   - `eval_batch_size`: Batch size for evaluation
   - `epochs`: Number of training epochs
   - `learning_rate`: Learning rate for training
   - `warmup_steps`: Number of warmup steps
   - `metric_for_best_model`: Metric to use for selecting the best model
   - `early_stopping_patience`: Number of evaluation calls with no improvement after which training will be stopped

You can also override these configurations using command-line arguments when running the script.

## DeepSpeed Integration

This project supports DeepSpeed for efficient training on multiple GPUs. To use DeepSpeed, make sure you have it installed and configured properly. The DeepSpeed configuration is stored in `ds_config.json`.

To run training with DeepSpeed, use the `--deepspeed` flag:

```
python main.py --mode train --deepspeed ds_config.json [OTHER OPTIONS]
```

## Customization

1. To use a different pre-trained model, modify the `model_name` in `config.yml` or pass it as a command-line argument.

2. To change the dataset, modify the `dataset` section in `config.yml` or use the corresponding command-line arguments.

3. For fine-tuning training parameters, adjust the `training` section in `config.yml` or use command-line arguments.

4. To modify the undertranslation or overtranslation logic, edit the corresponding functions in the `inference.py` file.
