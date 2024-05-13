import transformers
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import  Seq2SeqTrainer
import datasets
import matplotlib.pyplot as plt

import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

from datasets import Dataset, DatasetDict
import pandas as pd

# Load your CSV file using pandas
test_df = pd.read_csv('./test.csv')
train_df = pd.read_csv('./train.csv')
val_df = pd.read_csv('./valid.csv')

# Create a Dataset object
dataset = Dataset.from_pandas(train_df)
dataset2 = Dataset.from_pandas(val_df)
dataset3 = Dataset.from_pandas(test_df)

# Create a DatasetDict with a 'train' split
dataset_dict = DatasetDict({
    'train': dataset,
    'validation': dataset2,
    'test' : dataset3
})

max_input = 1024
max_target = 128
batch_size = 2
model_checkpoints = "facebook/bart-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
metric = load_metric('rouge')

def preprocess_data(examples):
    # #get all the dialogues
    inputs = [dialogue for dialogue in examples['text']]
    targets = [dialogue for dialogue in examples['summary']]

    # #tokenize the dialogues
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

    #tokenize the summaries
    labels = tokenizer(text_target=targets, max_length=max_target, padding='max_length', truncation=True)

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    #set labels
    model_inputs['labels'] = labels['input_ids']
    #return the tokenized data
    #input_ids, attention_mask and labels
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rouge(pred):
    predictions, labels = pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    #decode labels

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    #compute results
    res = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #get %
    res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

    pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    res['gen_len'] = np.mean(pred_lens)

    return  {k: round(v, 4) for k, v in res.items()}

tokenize_data = dataset_dict.map(preprocess_data, batched = True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

from transformers import DataCollatorForSeq2Seq
#collator to create batches. It preprocess data with the given tokenizer
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    'conversation-summ', #save directory
    evaluation_strategy='steps',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size= 4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=200,
    predict_with_generate=True,
    eval_accumulation_steps=1000,
    save_steps = 2000,
    eval_steps= 2000,
    fp16=True #available only with CUDA
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenize_data['train'],
    eval_dataset=tokenize_data['validation'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)

trainer.train()
