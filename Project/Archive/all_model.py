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


seed = 42
classes = 10
epochs = 2
learning_rate = 0.0001
batch_size = 32

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_base", use_fast=False)
    tokenizer.model_max_length = 512
    return tokenizer(examples["tweet"], padding="max_length", truncation=True)

def transform_labels(datapoint, part, num = 10):
    lb = np.array(datapoint["labels"])
    new_lb = torch.zeros(num)

    if part == 2:
        new_lb[lb + 3] = 1
    else:
        new_lb[lb] = 1

    return {
        "text" : datapoint["text"],
        "labels" : new_lb
    }


def combine_labels(ds, part):
    return ds.map(lambda datapoint : transform_labels(datapoint, part))


ds = load_dataset('masakhane/masakhanews', 'hau')
ds = ds.remove_columns(["text", "url", "headline"])
ds = ds.rename_column("label", "labels")
ds = ds.rename_column("headline_text", "text")

tok = combine_labels(ds, 2)
tok = tok.map(tokenize_function, batched=True)
tok = tok.remove_columns(["text"])
tok.set_format("torch")

train_dataset = tok["train"].shuffle(seed=seed)
test_dataset = tok["test"].shuffle(seed=seed)
val_dataset = tok["validation"].shuffle(seed=seed)

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
    #loss_fn=custom_loss,
)



# trainer.train()
# trainer.save_model("part_2_model")

#Evalualte the model on the test set
eval_results = trainer.evaluate()
print(eval_results)
#cohens kappa, precision, recall, f1 score, accuracy, hot or not 

