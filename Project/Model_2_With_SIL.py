# 
# !pip install --no-cache-dir transformers sentencepiece
# !pip install torch tensorflow
# !pip install datasets
# !pip install transformers[torch]
# !pip install accelerate -U
# !pip install evaluate


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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"""
HYPER PARAMS
"""
seed = 42
classes = 7
epochs = 1
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#Make the 2 models on the afriberta pre trained
student_model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)


#select the optimizer
student_optimizer = AdamW(student_model.parameters(), lr=learning_rate)


training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=student_optimizer,
    num_training_steps=training_steps,
    num_warmup_steps=0
)


student_model.to(device)


prog_bar = tqdm(range(training_steps))

student_model.train()

for epoch in range(epochs):

    teacher_model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)
    teacher_model.to(device)
    #copy the student model to be the teacher model
    teacher_model.load_state_dict(student_model.state_dict())
    
    #define optimizer & scheduler for teacher model
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=learning_rate)
    teacher_optimizer.zero_grad()
    training_steps_1 = epochs * len(train_dataloader)
    lr_scheduler_1 = get_scheduler(
        name="linear",
        optimizer=teacher_optimizer,
        num_training_steps=training_steps_1,
        num_warmup_steps=0
    )

    
    for batch in train_dataloader:
        #First train the teacher model 
        batch = {k : v.to(device) for k, v in batch.items()}
        teacher_model.train()
        outputs = teacher_model(**batch)
            
        loss = outputs.loss
        loss.backward()
        teacher_optimizer.step()
        lr_scheduler_1.step()
        teacher_optimizer.zero_grad()
        prog_bar.update(1)

        # Get teacher model predictions for the inputs next (new labels)

        with torch.no_grad():
            teacher_logits = teacher_model(**batch).logits

        # softmax the teacher logits for pseudo-labels
        pseudo_labels = softmax(teacher_logits, dim=1) #--> no argmax just use raw softmax

        # Train the student model using the teacher pseudo-labels
        student_model.train()
        student_optimizer.zero_grad()
        student_logits = student_model(**batch).logits


        loss = custom_loss(student_logits, pseudo_labels)
        loss.backward()
        
        student_optimizer.step()

    student_model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []

        for val_batch in val_dataloader:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            val_outputs = student_model(**val_batch)
            val_logits = val_outputs.logits

            # Assuming your labels are in the batch as 'labels'
            true_labels.extend(val_batch['labels'].cpu().numpy())
            
            # Calculate predictions (e.g., argmax of the logits)
            predictions.extend(val_logits.argmax(dim=1).cpu().numpy())

        #confusion matrix
        print(confusion_matrix(true_labels, predictions))
        # Calculate accuracy as an example evaluation metric
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Validation Accuracy: {accuracy}")

        
        # Put in eval mode
        #student_model.eval()



