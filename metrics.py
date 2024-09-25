import evaluate
import numpy as np

bleu_metric = evaluate.load('bleu')
bertscore_metric = evaluate.load('bertscore')
meteor_metric = evaluate.load('meteor')
rouge_metric = evaluate.load('rouge')


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bertscore = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, model_type='bert-base-uncased') 
    bertscore_f1 = sum(bertscore['f1']) / len(bertscore['f1'])
    
    return {
        'bleu': bleu['bleu'], 
        'bertscore': bertscore_f1, 
        'meteor': meteor['meteor'], 
        'rouge': rouge['rouge1']
    }