#IMPORTS
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
from sklearn.metrics import cohen_kappa_score, precision_recall_curve

"""
HYPER PARAMS
"""
seed = 42
classes = 10
epochs = 7
learning_rate = 5e-5
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_metrics(logits,labels):
    # logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels=np.argmax(labels, axis=-1)


    # Use evaluate.load to load pre-defined accuracy
    acc = evaluate.load("accuracy")
    accuracy = acc.compute(predictions=predictions, references=labels)

    # Use evaluate.load to load pre-defined F1, precision, and recall metrics
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    f1_value = f1.compute(predictions=predictions, references=labels, average='weighted')  # You can adjust 'average' if needed
    precision_value = precision.compute(predictions=predictions, references=labels, average='weighted')  # You can adjust 'average' if needed
    recall_value = recall.compute(predictions=predictions, references=labels, average='weighted')  # You can adjust 'average' if needed

    cohens_kappa=cohen_kappa_score(labels, predictions)


    return {
        "accuracy": accuracy['accuracy'],
        "f1": f1_value['f1'],
        "precision": precision_value['precision'],
        "recall": recall_value['recall'],
        "cohenkappa": cohens_kappa,
        'val loss':100000000,

    }



"""
Tokenize function, tokenizes the string data in the "text" field of the dataset
"""
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_base", use_fast=False)
    tokenizer.model_max_length = 512
    return tokenizer(examples["text"], padding="max_length", truncation=True)

"""
Function to edit the labels of the various datasets to have the same number of output classes
i.e. because we have 3 classes from the first dataset and 7 labels in the second, both labels should be 
edited to contain 10 classes 
""" 
# --> CHECK HERE
def one_hot_encode_labels(dataset, dataset_number, extend_classes = 10):
    orig_labels = np.array(dataset["labels"])
    new_labels = torch.zeros(extend_classes)

    if dataset_number == 1: #If were in dataset 1 , we need to add 7 zeros (classes) to the end
        new_labels[orig_labels] = 1 #Set the first 3 elements to the dataset 1 labels
    else: #If we're in dataset 2. we need to add 3 zeros (classes) to the beginning 
        new_labels[orig_labels + 3] = 1 #Set the last 7 elements to the dataset 2 labels
    
    #Return the original text with new one hot encoded labels
    return {
        "text" : dataset["text"],
        "labels" : new_labels
    }

"""
Function to map the original labels to the newly created one-hot labels
where the newly created one-hot lables have the length equal to the total number of classes
here: 3 classes from data set 1 + 7 classes from data set 2 = 10 classes in total
"""
def construct_labels(dataset, dataset_number):
    return dataset.map(lambda datapoint : one_hot_encode_labels(datapoint, dataset_number))


#First dataset is Igbo from the Naija Sentimient Twitter Dataset
ds = load_dataset("HausaNLP/NaijaSenti-Twitter", "ibo")


#Fix the column names to work with our functions
ds = ds.rename_column("label", "labels") 
ds = ds.rename_column("tweet", "text")

# Tokenize the data set
tokenized_datasets = construct_labels(ds, 1) 
tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

#Create train, validation and testing sets 
train_dataset_1 = tokenized_datasets["train"].shuffle(seed=seed)
test_dataset_1 = tokenized_datasets["test"].shuffle(seed=seed)
val_dataset_1 = tokenized_datasets["validation"].shuffle(seed=seed)


del ds #Remove the old dataset 
del tokenized_datasets #Remove the old tokenized set 
#Second dataset is Hausa from Masakhanews text classification dataset
ds = load_dataset('masakhane/masakhanews', 'hau')

#Fix the column names to work with our functions
ds = ds.remove_columns(["text", "url", "headline"])
ds = ds.rename_column("label", "labels")
ds = ds.rename_column("headline_text", "text")

#tokenize the data set
tokenized_datasets = construct_labels(ds, 2)
tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

#Create train, validation and testing sets 
train_dataset_2 = tokenized_datasets["train"].shuffle(seed=seed)
test_dataset_2 = tokenized_datasets["test"].shuffle(seed=seed)
val_dataset_2 = tokenized_datasets["validation"].shuffle(seed=seed)

del ds #Remove the old dataset 
del tokenized_datasets #Remove the original tokenized sets


"""
Now we need to train model 1 on the first dataset 
"""
model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes).to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_dataset_1, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset_1, batch_size=8)
num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


model.eval()
true_labels = []
predicted_labels = []

for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))

print("===========MODEL 1 ON DATASET 1 REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==================================================")


"""
Train on the second dataset
"""
del train_dataloader
del val_dataloader
model.train()
train_dataloader = DataLoader(train_dataset_2, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset_2, batch_size=8)
num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.eval()
true_labels = []
predicted_labels = []

for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))

print("===========MODEL 2 ON DATASET 2 REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==================================================")


model.eval()
true_labels = []
predicted_labels = []
del val_dataloader
val_dataloader = DataLoader(val_dataset_1, batch_size=8)
for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1) #Unsure about this - needs to be soft labels ? 
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))

print("===========MODEL 2 ON DATASET 1 REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==================================================")


"""
==========MODEL 1 ON DATASET 1 REPORT===========
Accuracy: 0.7876154263986963
F1 Score: 0.7874910200927531
Precision: 0.7873872964342025
recall: 0.7876154263986963
Cohen's Kappa: 0.6724382640066656
==================================================

===========MODEL 2 ON DATASET 2 REPORT===========
Accuracy: 0.8391167192429022
F1 Score: 0.8263375065379148
Precision: 0.8309560769078235
recall: 0.8391167192429022
Cohen's Kappa: 0.8108724643784656
==================================================

===========MODEL 2 ON DATASET 1 REPORT===========
Accuracy: 0.12819120043454643
F1 Score: 0.16484230972431413
Precision: 0.23084239158711428
recall: 0.12819120043454643
Cohen's Kappa: 0.09545244291407817
==================================================
"""