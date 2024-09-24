import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

config_dataset = config['dataset']
config_train = config['training'] 
config_model = config['model']
config_tokenizer = config['tokenizer']

metric_greater_is_better_map = {
    'bleu': True,
    'bertscore': True,
    'meteor': True,
    'rouge': True,
    'loss': False
}

metric_greater_is_better = metric_greater_is_better_map[config_train['metric_for_best_model']]

en_token = config_tokenizer['tokens']['en']
ru_token = config_tokenizer['tokens']['ru']
device = config['device']