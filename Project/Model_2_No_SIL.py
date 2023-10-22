#IMPORTS
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch import nn
from tqdm.auto import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter
from zmq import device

from Config_Manager import get_dataset, compute_metrics, SEED, CLASSES, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE


"""
HYPER PARAMS FROM CONFIG FILE
"""
seed = SEED
classes = CLASSES
epochs = EPOCHS
learning_rate = LEARNING_RATE
batch_size = BATCH_SIZE
device = DEVICE



dataset = get_dataset("masakhane")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
val_dataset = dataset["val"]
del dataset

#Load in the model that was saved before
model = AutoModelForSequenceClassification.from_pretrained("model1").to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)


model.train()
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
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
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
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

print("===========MODEL 2 ON DATASET 1 REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==================================================")