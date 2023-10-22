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

train_set_naija = get_dataset("naija")["test"]
train_set_masakhane = get_dataset("masakhane")["test"]

test_1_dataloader = DataLoader(train_set_naija, batch_size=batch_size)
test_2_dataloader = DataLoader(train_set_masakhane, batch_size=batch_size)

#Load in the model that was saved before
model_1 = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
model_2_no_sil = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_2_No_SIL").to(device)
# model_2_sil = AutoModelForSequenceClassification.from_pretrained("model_2_SIL").to(device)

model_1.config.loss_name = "cross_entropy" #use cross entropy loss function
model_2_no_sil.config.loss_name = "cross_entropy" #use cross entropy loss function
# model_2_sil.config.loss_name = "cross_entropy" #use cross entropy loss function

model_1.eval()
#Predictions for Model 1 on Dataset 1:
true_labels = []
predicted_labels = []
for batch in test_1_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model_1(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))
print("===========MODEL 1 ON DATASET 1 (NaijaSenti) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("=============================================================\n\n")


#Predictions for Model 2 on Dataset 2:
model_2_no_sil.eval()
true_labels = []
predicted_labels = []

for batch in test_2_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model_2_no_sil(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))
print("===========MODEL 2 ON DATASET 2 (Masakhane) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("=============================================================\n\n")


#Predictions for Model 2 on Dataset 1:
model_2_no_sil.eval()
true_labels = []
predicted_labels = []

for batch in test_1_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model_2_no_sil(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    true = batch["labels"]

    true_labels.extend(true.cpu().numpy())
    predicted_labels.extend(logits.cpu().numpy())

custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))
print("===========MODEL 2 ON DATASET 1 (NaijaSenti) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("=============================================================\n\n")
