# model_name='results/opus-mt-en-ru/checkpoint-300'
model_name='Helsinki-NLP/opus-mt-en-ru'
# input_text='I am working on a project at Apple'
input_text='I am eating an apple'


python main.py --mode inference --model_name "$model_name" --input_text "$input_text" --translation_mode "undertranslation"