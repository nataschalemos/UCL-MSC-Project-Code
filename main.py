"""
This script is used to train and evaluate a model for multimodal emotion recognition.

Notes:
Loss function: large-margin softmax loss function
(adapted from: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#largemarginsoftmaxloss)
"""

# Imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

from pytorch_metric_learning import losses
import wandb

from dataset_loader import IemocapDataset, ToTensor
from train_function import fit
from validation_function import validate
from model_structure import newbob, Fusion
from utils import collate_batch

from fairseq.models.roberta import RobertaModel

"""
Variables to define - change the variables below accordingly (~ should only have to change wdir)
"""

# Define path to directory with files
wdir = sys.argv[1]
# Define directory with data
data_dir = sys.argv[2]
# Define directory with downloaded models
models_dir = sys.argv[3]

# # Define path to directory with files
# wdir = '/Volumes/TOSHIBA EXT/Code/'
# # Define directory with data
# data_dir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'
# # Define directory with downloaded models
# models_dir = wdir + 'Models/bert_kmeans/'

# Define path to "labels_train_t.txt" file
labels_file = data_dir + 'labels_train_t.txt'

# Define path to directory with TSB and TAB input embeddings
root_dir_Roberta = data_dir + 'GPT2Tokens'
root_dir_SpeechBERT = data_dir + 'vqw2vTokens'
root_dir_TAB = data_dir + 'FullBertEmb'
root_dir_text = data_dir + 'Text'
root_dir_audio = data_dir + 'Audio'

# Change current working directory
os.chdir(wdir)
print("Current working directory: {0}".format(os.getcwd()))

"""
Train model
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# Define learning parameters
config = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 5e-5,
    "optimizer": optim.Adam,
    "scheduler": newbob,
    "factor": 0.5,
    "criterion": losses.LargeMarginSoftmaxLoss(num_classes=4,
                                               embedding_size=4,
                                               margin=4,
                                               scale=1).to(device)
}

# Start a W&B run
wandb.init(project="run-s5_new_model_v1", entity="natascha-msc-project", config=config)
# Save model inputs and hyperparameters
wandb.config

# Split data into training and validation data
label_files = pd.read_csv(labels_file, header=None, delimiter='\t')

train_label_files = label_files[label_files[0].str.contains('s_1|s_2|s_3|s_4')]
val_label_files = label_files[label_files[0].str.contains('s_5')]

# Load dataset and dataloader
train_dataset = IemocapDataset(labels_file=train_label_files,
                               dir=data_dir,
                               max_text_tokens=100,
                               max_audio_tokens=310,
                               transform=ToTensor())

val_dataset = IemocapDataset(labels_file=val_label_files,
                             dir=data_dir,
                             max_text_tokens=100,
                             max_audio_tokens=310,
                             transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, collate_fn=collate_batch)

val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_batch)

# Load sub-models
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
speechBert = RobertaModel.from_pretrained(models_dir,checkpoint_file='bert_kmeans.pt')
for param in speechBert.parameters():
    param.requires_grad = False

# Instantiate model class
model = Fusion(roberta, speechBert).to(device)

# Instantiate optimizer class
optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
# Instantiate learning rate scheduler class
scheduler = config["scheduler"](optimizer, factor=config["factor"], model=model)

# Lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []

# initialize step for WandB plots
step_train = 0
step_val = 0

# Train the model
for epoch in range(config["epochs"]):
    print(f'Epoch {epoch + 1} of {config["epochs"]}')

    # Fit model
    train_epoch_loss, train_epoch_accuracy, curr_train_step = fit(model, train_dataloader, train_dataset, optimizer,
                                                                  config["criterion"], device, step_train)
    step_train += curr_train_step

    # Validate model
    val_epoch_loss, val_epoch_accuracy = validate(model, val_dataloader, val_dataset, config["criterion"], device,
                                                  step_val)
    step_val += 1

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')

    # Update learning rate scheduler
    scheduler.step(val_epoch_accuracy)


"""
Save model
"""

# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), data_dir + "outputs/model.pth")

print('TRAINING COMPLETE')
