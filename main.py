import argparse
from inference import inference
from params import default_config as config
import torch
import subprocess
import yaml
def main():
    parser = argparse.ArgumentParser(description='English to Russian Translation Model')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True, help='Mode of operation')
    
    parser.add_argument('--dataset_name', default=config['dataset']['name'], help=f'Dataset name (default: {config["dataset"]["name"]})')
    parser.add_argument('--dataset_subset', default=config['dataset']['subset'], help=f'Dataset subset (default: {config["dataset"]["subset"]})')
    
    parser.add_argument('--model_name', default=config['model']['model_name'], help=f'Model name (default: {config["model"]["model_name"]})')
    parser.add_argument('--model_max_length', type=int, default=config['model']['max_length'], help=f'Model max length (default: {config["model"]["max_length"]})')
    
    parser.add_argument('--training_batch_size', type=int, default=config['training']['training_batch_size'], help=f'Training batch size (default: {config["training"]["training_batch_size"]})')
    parser.add_argument('--eval_batch_size', type=int, default=config['training']['eval_batch_size'], help=f'Evaluation batch size (default: {config["training"]["eval_batch_size"]})')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help=f'Number of epochs (default: {config["training"]["epochs"]})')
    parser.add_argument('--learning_rate', type=float, default=config['training']['learning_rate'], help=f'Learning rate (default: {config["training"]["learning_rate"]})')
    parser.add_argument('--warmup_steps', type=int, default=config['training']['warmup_steps'], help=f'Warmup steps (default: {config["training"]["warmup_steps"]})')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config['training']['gradient_accumulation_steps'], help=f'Gradient accumulation steps (default: {config["training"]["gradient_accumulation_steps"]})')
    parser.add_argument('--max_grad_norm', type=float, default=config['training']['max_grad_norm'], help=f'Max gradient norm (default: {config["training"]["max_grad_norm"]})')
    parser.add_argument('--weight_decay', type=float, default=config['training']['weight_decay'], help=f'Weight decay (default: {config["training"]["weight_decay"]})')
    parser.add_argument('--logging_steps', type=int, default=config['training']['logging_steps'], help=f'Logging steps (default: {config["training"]["logging_steps"]})')
    parser.add_argument('--eval_steps', type=int, default=config['training']['eval_steps'], help=f'Evaluation steps (default: {config["training"]["eval_steps"]})')
    parser.add_argument('--save_steps', type=int, default=config['training']['save_steps'], help=f'Save steps (default: {config["training"]["save_steps"]})')
    parser.add_argument('--metric_for_best_model', default=config['training']['metric_for_best_model'], help=f'Metric for best model (default: {config["training"]["metric_for_best_model"]})')
    parser.add_argument('--early_stopping_patience', type=int, default=config['training']['early_stopping_patience'], help=f'Early stopping patience (default: {config["training"]["early_stopping_patience"]})')
    parser.add_argument('--early_stopping_threshold', type=float, default=config['training']['early_stopping_threshold'], help=f'Early stopping threshold (default: {config["training"]["early_stopping_threshold"]})')
    parser.add_argument('--deepspeed', default=config['training']['deepspeed'], help=f'DeepSpeed config file (default: {config["training"]["deepspeed"]})')
    parser.add_argument('--translation_mode', choices=['normal', 'undertranslation', 'overtranslation'], default='normal', required=False, help='Translation mode to do inference (Defoult: normal)')
    parser.add_argument('--input_text', help='Text to translate (for inference mode only)')

    args = parser.parse_args()

    for key, value in vars(args).items():
        if key in config['dataset']:
            config['dataset'][key] = value
        elif key in config['model']:
            config['model'][key] = value
        elif key in config['training']:
            config['training'][key] = value

    with open('config.yml', 'w') as f:
        yaml.dump(config, f)

    if args.mode == 'train':
        num_available_gpus = torch.cuda.device_count()
        subprocess.run(['deepspeed', 'train.py', '--num_gpus', str(num_available_gpus)])
    elif args.mode == 'inference':
        if not args.input_text:
            parser.error('--input_text is required for inference mode')
        translated_text = inference(args.input_text, args.translation_mode)
        print(f'Original text:\n{args.input_text}\n')
        print(f'Translated text ({args.translation_mode} mode):\n{translated_text}')

if __name__ == '__main__':
    main()