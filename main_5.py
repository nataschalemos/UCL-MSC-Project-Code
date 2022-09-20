"""
*remove*

JUST FOR ABLATION STUDIES: CONTEXT WINDOWS

- TEST SET: SESSION 5
- CONTEXT WINDOW: [-2,0]

-------------------------------------------------------------------------------------
This script is used to train and evaluate a model for multimodal emotion recognition.
For new implementation of dataset loader

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
from torch.nn import MultiMarginLoss, L1Loss
import sys
import torch.nn.functional as F
from torchvision import models

from pytorch_metric_learning import losses
import wandb

from dataset_loader import IemocapDataset, ToTensor
from train_function import fit
from validation_function import validate
from test_function import test
from model_structure import newbob, Fusion, LHS
from utils import collate_tokens, collater, get_num_sentences, collate_batch
from finetune_ssl import finetune

from fairseq.models.roberta import RobertaModel

# Define seed for reproducibility
seed = 2022
torch.manual_seed(seed)

"""
Variables to define - change the variables below accordingly (~ should only have to change wdir)
"""

# Define path to directory with files
wdir = sys.argv[1]
# Define directory with data
data_dir = sys.argv[2]
# Define directory with downloaded models
models_dir = sys.argv[3]
# Define window for BERT sentence context embeddings (0)
cw = sys.argv[4]
# Define max sequence length for text tokens
max_text_tokens = sys.argv[5]
# Define max sequence length for audio tokens
max_audio_tokens = sys.argv[6]

# # Define path to directory with files
# wdir = '/Volumes/TOSHIBA EXT/Code/'
# # Define directory with data
# data_dir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'
# # Define directory with downloaded models
# models_dir = wdir + 'Models/bert_kmeans/'
# # Define window for BERT sentence context embeddings
context_window = 0
# # Define max sequence length for text tokens
# max_text_tokens = 256
# # Define max sequence length for audio tokens
# max_audio_tokens = 1024

# Define path to "labels_train_t.txt" file
labels_file = data_dir + 'labels_train_t.txt'

# Define path to directory with TSB and TAB input embeddings
root_dir_Roberta = data_dir + 'GPT2Tokens'
root_dir_SpeechBERT = data_dir + 'vqw2vTokens'
root_dir_TAB = data_dir + 'BertEmb'
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

# Define loss functions and parameters
large_margin_softmax_loss = losses.LargeMarginSoftmaxLoss(num_classes=4,
                                                          embedding_size=4,
                                                          margin=4,
                                                          scale=1).to(device)

multi_margin_loss = MultiMarginLoss(p=1, margin=0.4, weight=None, size_average=None, reduce=None, reduction='mean')

f1_loss = L1Loss(reduction='sum')  # Note: when using this loss need to argmax predictions in train and val functions

# Define learning parameters
config = {
    "batch_size": 16,
    "epochs": 100,
    "patience": 10,
    "learning_rate": 5e-5,
    "optimizer": optim.Adam,
    "scheduler": newbob,
    "factor": 0.5,
    "criterion": multi_margin_loss.to(device),
    "context_window": int(context_window),
    "freeze_models": True,
    "penalty": False,
    "finetune": False,
    "log_results": True
}

# Define learning parameters for finetuning
config_finetune = {
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 5e-5,
    "optimizer": optim.Adam,
    "scheduler": newbob,
    "factor": 0.5,
    "criterion": multi_margin_loss.to(device),
    "context_window": int(context_window),
    "freeze_models": False,
    "penalty": False,
    "log_results": False
}

# Start a W&B run
wandb.init(project="AS_setup_1_context_past_2", entity="natascha-msc-project", config=config)
# Save model inputs and hyperparameters
wandb.config

# Split data into training and validation data
label_files = pd.read_csv(labels_file, header=None, delimiter='\t')

full_train_label_files = label_files[label_files[0].str.contains('s_1|s_2|s_3|s_4')]
# full_train_label_files = label_files[label_files[0].str.contains('s_1')]
val_label_files = full_train_label_files.sample(frac=0.1, random_state=seed)
train_label_files = full_train_label_files.drop(val_label_files.index)

test_label_files = label_files[label_files[0].str.contains('s_5')]

# Print number of sentences per emotion in each train/val set
train_num = get_num_sentences(train_label_files)
val_num = get_num_sentences(val_label_files)
print("\nNumber of sentences per emotion: ")
print("Train dataset: angry = {}, happy/excited = {}, sad = {}, neutral = {}".format(train_num[0], train_num[1],
                                                                                     train_num[2], train_num[3]))
print("Val dataset: angry = {}, happy/excited = {}, sad = {}, neutral = {}".format(val_num[0], val_num[1], val_num[2],
                                                                                   val_num[3]))

# Store number of utterances in each set for this cv split
wandb.run.summary["train_utterances"] = len(train_label_files)
wandb.run.summary["val_utterances"] = len(val_label_files)
wandb.run.summary["test_utterances"] = len(test_label_files)

# Load dataset and dataloader
train_dataset = IemocapDataset(labels_file=train_label_files,
                               dir=data_dir,
                               context_window=context_window,
                               device=device,
                               max_text_tokens=int(max_text_tokens),
                               max_audio_tokens=int(max_audio_tokens),
                               transform=ToTensor())

val_dataset = IemocapDataset(labels_file=val_label_files,
                             dir=data_dir,
                             context_window=context_window,
                             device=device,
                             max_text_tokens=int(max_text_tokens),
                             max_audio_tokens=int(max_audio_tokens),
                             transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, collate_fn=collate_batch)

val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_batch)

# Load sub-models
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
# speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt')

if config["finetune"]:
    "Jointly fine-tuning SSL models..."
    # Finetune sub-models
    finetune(data_dir, models_dir, config_finetune, train_dataset, val_dataset, train_dataloader, val_dataloader, device)

    # Load sub-models
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt')

    # Load finetuned sub-model weights
    "Loading fine-tuned models"
    model = LHS(roberta, speechBert, freeze_models=config["freeze_models"]).to(device) # TODO: check if alright to freeze these layers in this step and not later
    model.load_state_dict(torch.load(data_dir + "outputs/fine_tuned_LHS_branch.pth"))

    # Build full model using fine-tuned weights # TODO: select layers for b1 and b2 in fine-tuned model and set these as b1 and b2 in the Fusion model
    roberta = model.b1
    speechBert = model.b2

else:
    # Load sub-models
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt')

# Instantiate model class
model = Fusion(roberta, speechBert, freeze_models=config["freeze_models"]).to(device)

# Instantiate optimizer class
optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
# Instantiate learning rate scheduler class
scheduler = config["scheduler"](optimizer, factor=config["factor"], model=model)

# Lists to store per-epoch loss and accuracy values
train_loss, train_u_accuracy, train_w_accuracy = [], [], []
val_loss, val_u_accuracy, val_w_accuracy = [], [], []

# Early stopping parameters
last_best_loss = 1000
trigger_times_best = 0

# initialize step for WandB plots
step_train = 0
step_val = 0

# Train the model
for epoch in range(config["epochs"]):
    print(f'Epoch {epoch + 1} of {config["epochs"]}')

    # Fit model
    model.train()
    train_epoch_loss, train_epoch_u_accuracy, train_epoch_w_accuracy, curr_train_step = fit(model, train_dataloader,
                                                                                            train_dataset,
                                                                                            optimizer,
                                                                                            config["criterion"], device,
                                                                                            step_train,
                                                                                            config["penalty"],
                                                                                            config["log_results"])
    step_train += curr_train_step

    # Validate model
    val_epoch_loss, val_epoch_u_accuracy, val_epoch_w_accuracy = validate(model, val_dataloader, val_dataset, config["criterion"], device, step_val, config["log_results"])
    step_val += 1

    train_loss.append(train_epoch_loss)
    train_u_accuracy.append(train_epoch_u_accuracy)
    train_w_accuracy.append(train_epoch_w_accuracy)
    val_loss.append(val_epoch_loss)
    val_u_accuracy.append(val_epoch_u_accuracy)
    val_w_accuracy.append(val_epoch_w_accuracy)

    print(f"Train Loss: {train_epoch_loss:.4f}, Train UA: {train_epoch_u_accuracy:.2f}, Train WA: {train_epoch_w_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val UA: {val_epoch_u_accuracy:.2f}, Val WA: {val_epoch_w_accuracy:.2f}')

    # Early stopping
    if val_epoch_loss > last_best_loss:
        trigger_times_best += 1
        if trigger_times_best >= config["patience"]:
            print('\nEarly stopping! \n loss has not improved from lowest loss.')
            break
    else:
        trigger_times_best = 0
        last_best_loss = val_epoch_loss

    # Update learning rate scheduler
    #scheduler.step(val_epoch_accuracy)

"""
Save model
"""

# # serialize the model to disk
# print('Saving model...')
# torch.save(model.state_dict(), data_dir + "outputs/fine_tuned_LHS_branch.pth")

print('TRAINING COMPLETE')

"""
Test final model
"""

# Load dataset and dataloader
test_dataset = IemocapDataset(labels_file=test_label_files,
                              dir=data_dir,
                              context_window=context_window,
                              device=device,
                              max_text_tokens=int(max_text_tokens),
                              max_audio_tokens=int(max_audio_tokens),
                              transform=ToTensor())

test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=0, collate_fn=collate_batch)

test_loss, test_unweighted_accuracy, test_weighted_accuracy, test_class_accuracy = test(model, test_dataloader, test_dataset, config["criterion"], device)

print(f"Test Loss: {test_loss:.4f}, Test UA: {test_unweighted_accuracy:.2f}, Test WA: {test_weighted_accuracy:.2f}")
wandb.run.summary({"test_unweighted_accuracy": test_unweighted_accuracy, "test_weighted_accuracy": test_weighted_accuracy, "test_loss": test_loss})