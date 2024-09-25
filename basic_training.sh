model_name='Helsinki-NLP/opus-mt-en-ru'


python main.py --mode train --model_name "$model_name" --metric_for_best_model "meteor" --early_stopping_patience 30 