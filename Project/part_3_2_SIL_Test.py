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
from torch.optim import AdamW 
from torch.nn.functional import softmax


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
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def custom_loss(predictions, labels):
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(predictions, labels)

def get_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)

    # Calculate accuracy
    correct = (predicted_classes == labels).sum().item()
    accuracy = correct / labels.size(0)  # Calculate accuracy as a fraction
    return accuracy

#Dataset --> label,text

ds = load_dataset('masakhane/masakhanews', 'hau') 
ds = ds.remove_columns(["text", "url", "headline"])
ds = ds.rename_column("label", "labels")
ds = ds.rename_column("headline_text", "text")

tokenized_datasets = ds.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
test_dataset = tokenized_datasets["test"].shuffle(seed=seed)
val_dataset = tokenized_datasets["validation"].shuffle(seed=seed)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)

optimizer = AdamW(model.parameters(), lr=learning_rate)


training_steps = epochs * len(train_dataloader) 
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=training_steps,
    num_warmup_steps=0
)

device = torch.device("cpu")
model.to(device)

prog_bar = tqdm(range(training_steps))

model.train()

for epoch in range(epochs):
    for batch in train_dataloader:
        batch = {k : v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        prog_bar.update(1)


