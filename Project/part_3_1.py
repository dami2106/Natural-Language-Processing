from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from  datasets  import  load_dataset
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch import nn
from tqdm.auto import tqdm
import evaluate


"""
HYPER PARAMS
"""
seed = 42
classes = 7
epochs = 3
learning_rate = 0.0001
batch_size = 32


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_base", use_fast=False)
    tokenizer.model_max_length = 512
    return tokenizer(examples["headline_text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

ds = load_dataset('masakhane/masakhanews', 'hau') 
# ds = load_dataset("HausaNLP/NaijaSenti-Twitter", "ibo")
tokenized_datasets = ds.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
test_dataset = tokenized_datasets["test"].shuffle(seed=seed)
val_dataset = tokenized_datasets["validation"].shuffle(seed=seed)

model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)
model.config.loss_name = "cross_entropy"

training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    seed = seed,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()

model.save(model.state_dict(), "classificiation_model.pt")

# model.load_state_dict(torch.load("model.pt"))