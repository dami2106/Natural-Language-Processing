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



dataset = get_dataset("masakhane")
train_dataset = dataset["train"]
val_dataset = dataset["val"]
del dataset

#Load in the model that was saved before
model = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)


model.train()
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

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

#Save the model to disk
model.save_pretrained("Saved_Models/model_2_No_SIL")