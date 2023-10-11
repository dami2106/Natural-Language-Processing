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
    return tokenizer(examples["headline_text"], padding="max_length", truncation=True)

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


ds = load_dataset('masakhane/masakhanews', 'hau') 
tokenized_datasets = ds.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
test_dataset = tokenized_datasets["test"].shuffle(seed=seed)
val_dataset = tokenized_datasets["validation"].shuffle(seed=seed)



#Make the 2 models on the afriberta pre trained
student_model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)
teacher_model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes)

#Select the optimizer
optimizer = AdamW(student_model.parameters(), lr=learning_rate)  # Adjust optimizer and learning rate as needed

student_training_args = TrainingArguments(
    output_dir="student_model",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    seed=seed,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
)

student_trainer = Trainer(
    model=student_model,
    args=student_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

#pre train the student 
student_trainer.train()

# Create a DataLoader 
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



for epoch in range(epochs):
    #copy the student model to be the teacher model
    teacher_model.load_state_dict(student_model.state_dict())

    # for batch in train_dataloader:
    inputs = train_dataset["input_ids"]
    labels = train_dataset["label"]

    # Get teacher model predictions for the inputs next
    with torch.no_grad():
        teacher_logits = teacher_model(inputs).logits

    # softmax the teacher logits for pseudo-labels
    pseudo_labels = torch.argmax(softmax(teacher_logits, dim=1), dim=1)

    # Train the student model using the above
    student_model.train()
    optimizer.zero_grad()
    student_logits = student_model(inputs).logits
    loss = custom_loss(student_logits, pseudo_labels)
    loss.backward()
    optimizer.step()

    # Evaluate the student model on the validation dataset
    student_model.eval()
    with torch.no_grad():
        student_predictions = []
        # for batch in val_dataloader:
        inputs = val_dataset["input_ids"]
        labels = val_dataset["label"]
        logits = student_model(inputs).logits
        predictions = torch.argmax(logits, dim=1)
        student_predictions.extend(predictions.cpu().numpy())

    # using function from above
    student_accuracy = get_accuracy(student_predictions, val_dataset["label"])

    # Print or log the validation accuracy for this epoch
    print(f"Epoch {epoch + 1}: Student Model Validation Accuracy: {student_accuracy}")

# Save the final model
# student_model.save_pretrained("student_model")


