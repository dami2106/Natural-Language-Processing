#IMPORTS
from transformers import AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from Config_Manager import get_dataset, SEED, CLASSES, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE
import sys 

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
val_dataset_d1 = get_dataset("naija")["val"]
del dataset

#Load in the model that was saved before
model = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
val_dataloader_d1 = DataLoader(val_dataset_d1, batch_size=batch_size)

num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

train_epoch_loss = []
val_epoch_loss = []
val_epoch_loss_d1 = []

progress_bar = tqdm(range(num_training_steps))
for epoch in range(epochs):
    step_loss = []
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step_loss.append(loss.item())

        progress_bar.update(1)
    
    train_epoch_loss.append(np.mean(step_loss))

    #Evaluate the model on the validation set
    model.eval()
    step_loss = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item())

    val_epoch_loss.append(np.mean(step_loss))

    model.eval()
    step_loss = []
    for batch in val_dataloader_d1:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item())

    val_epoch_loss_d1.append(np.mean(step_loss))


loss_data = [
    train_epoch_loss,
    val_epoch_loss,
    val_epoch_loss_d1
]

loss_data = np.array(loss_data)

np.save(f"Saved_Models/model_2_No_SIL/Model_2_No_SIL_Loss_{sys.argv[1]}.npy", loss_data)

# plt.plot(train_epoch_loss, label='Training Loss')
# plt.plot(val_epoch_loss,label='Validation Loss Dataset 2')
# plt.plot(val_epoch_loss_d1,label='Validation Loss Dataset 1')
# plt.legend()
# plt.xticks(np.arange(0, len(train_epoch_loss), 1))
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model 2 (No SIL) Loss")
# plt.savefig("Model_2_No_Sil_Loss.png", dpi = 300)

# #Save the model to disk
# model.save_pretrained("Saved_Models/model_2_No_SIL")