#IMPORTS
from transformers import AutoTokenizer
import pandas as pd
from  datasets  import  load_dataset
import numpy as np
import evaluate
import torch
import evaluate
from sklearn.metrics import cohen_kappa_score


"""
HYPER PARAMS FOR MODELS
=======================
"""
SEED = 42
CLASSES = 10
EPOCHS = 1
LEARNING_RATE = 5e-5
BATCH_SIZE = 128
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
"""
=======================
"""




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


"""
A funtion to get the tokenized formatted dataset.
PARAMS : dataset - the dataset to be tokenized (either naija or masakhane)
            seed - the seed to be used for shuffling the dataset
RETURN : a dictionary containing the train, test and validation sets
"""
def get_dataset(dataset, seed = 42):
    tokenized_datasets = None
    if dataset == "naija":
        #First dataset is Igbo from the Naija Sentimient Twitter Dataset
        ds = load_dataset("HausaNLP/NaijaSenti-Twitter", "ibo")

        #Fix the column names to work with our functions
        ds = ds.rename_column("label", "labels") 
        ds = ds.rename_column("tweet", "text")

        # Tokenize the data set
        tokenized_datasets = construct_labels(ds, 1) 

    else:
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
    train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    test_dataset = tokenized_datasets["test"].shuffle(seed=seed)
    val_dataset = tokenized_datasets["validation"].shuffle(seed=seed)

    return {
        "train" : train_dataset,
        "test" : test_dataset,
        "val" : val_dataset
    }


"""
A function to compute the metrics of the model
PARAMS : logits - the logits of the model
         labels - the labels of the model
RETURN : a dictionary containing the metrics of the model
"""
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