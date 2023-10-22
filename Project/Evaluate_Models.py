#IMPORTS
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from Config_Manager import get_dataset, compute_metrics, evaluate_model, SEED, CLASSES, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE

"""
HYPER PARAMS FROM CONFIG FILE
"""
batch_size = BATCH_SIZE
device = DEVICE

train_set_naija = get_dataset("naija")["test"]
train_set_masakhane = get_dataset("masakhane")["test"]

test_1_dataloader = DataLoader(train_set_naija, batch_size=batch_size)
test_2_dataloader = DataLoader(train_set_masakhane, batch_size=batch_size)

#Load in the model that was saved before
model_1 = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
model_2_no_sil = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_2_No_SIL").to(device)
model_2_sil = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_2_SIL").to(device)

model_1.config.loss_name = "cross_entropy" #use cross entropy loss function
model_2_no_sil.config.loss_name = "cross_entropy" #use cross entropy loss function
model_2_sil.config.loss_name = "cross_entropy" #use cross entropy loss function

custom_metrics_dict = evaluate_model(model_1, test_1_dataloader)
print("===========MODEL 1 ON DATASET 1 (NaijaSenti) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==============================================================\n\n")


custom_metrics_dict = evaluate_model(model_2_no_sil, test_2_dataloader)
print("===========NO SIL MODEL ON DATASET 2 (Masakhane) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("==================================================================\n\n")


#Predictions for Model 2 on Dataset 1:

custom_metrics_dict = evaluate_model(model_2_no_sil, test_1_dataloader)
print("===========NO SIL MODEL ON DATASET 1 (NaijaSenti) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("===================================================================\n\n")


"""
SIL MODEL 
"""

#Predictions for SIL Model 2 on Dataset 2:
custom_metrics_dict = evaluate_model(model_2_sil, test_2_dataloader)
print("===========SIL MODEL ON DATASET 2 (Masakhane) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("===============================================================\n\n")


#Predictions for SIL Model 2 on Dataset 1:
custom_metrics_dict = evaluate_model(model_2_sil, test_1_dataloader)
print("===========SIL MODEL ON DATASET 1 (NaijaSenti) REPORT===========")
print("Accuracy:", custom_metrics_dict["accuracy"])
print("F1 Score:", custom_metrics_dict["f1"])
print("Precision:", custom_metrics_dict["precision"])
print("recall:", custom_metrics_dict["recall"])
print("Cohen's Kappa:", custom_metrics_dict["cohenkappa"])
print("================================================================\n\n")
