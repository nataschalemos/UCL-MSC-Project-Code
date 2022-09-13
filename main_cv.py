"""
This script is used to train and evaluate a model for multimodal emotion recognition.
Using k-fold cross-validation to train and evaluate model

Notes:
Loss function: large-margin softmax loss function
(adapted from: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#largemarginsoftmaxloss)
"""

# Imports
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import MultiMarginLoss, L1Loss
import sys
import torch.nn.functional as F

from pytorch_metric_learning import losses
import wandb

from dataset_loader import IemocapDataset, ToTensor
from train_function import fit
from validation_function import validate
from model_structure import newbob, Fusion
from utils import collate_tokens, collater, get_num_sentences, collate_batch
from cross_validation import KfoldCv

from fairseq.models.roberta import RobertaModel

# Define seed for reproducibility
seed = 2022
torch.manual_seed(seed)

"""
Variables to define - change the variables below accordingly (~ should only have to change wdir)
"""

#TODO uncomment

# Define path to directory with files
wdir = sys.argv[1]
# Define directory with data
data_dir = sys.argv[2]
# Define directory with downloaded models
models_dir = sys.argv[3]
# Define window for BERT sentence context embeddings (1, 2 or 3)
context_window = sys.argv[4]
# Define max sequence length for text tokens
max_text_tokens = sys.argv[5]
# Define max sequence length for audio tokens
max_audio_tokens = sys.argv[6]

# TODO: comment

# # Define path to directory with files
# wdir = '/Volumes/TOSHIBA EXT/Code/'
# # Define directory with data
# data_dir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'
# # Define directory with downloaded models
# models_dir = wdir + 'Models/bert_kmeans/'
# # Define window for BERT sentence context embeddings
# context_window = 3
# # Define max sequence length for text tokens
# max_text_tokens = 256
# # Define max sequence length for audio tokens
# max_audio_tokens = 1024

# Define path to "labels_train_t.txt" file
labels_file = data_dir + 'labels_train_t.txt'

# Define path to directory with TSB and TAB input embeddings
root_dir_Roberta = data_dir + 'GPT2Tokens'
root_dir_SpeechBERT = data_dir + 'vqw2vTokens'
root_dir_TAB = data_dir + 'FullBertEmb' + str(context_window)
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
    "freeze_models": True
}

# # Start a W&B run
# wandb.init(project="run-new-model-cv-1", group="exp_2", entity="natascha-msc-project", config=config)
# # Save model inputs and hyperparameters
# wandb.config

# Split data into training and validation data
label_files = pd.read_csv(labels_file, header=None, delimiter='\t')

sessions = ['s_1', 's_2', 's_3', 's_4', 's_5']

# # Load sub-models
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
# speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt')

# Perform k-fold cross-validation
#train_UA_runs,train_WA_runs,val_UA_runs,val_WA_runs,test_UA_runs,test_WA_runs = KfoldCv(5, seed, config, label_files, sessions, max_text_tokens, max_audio_tokens, device, data_dir, roberta, speechBert)
train_UA_runs,train_WA_runs,val_UA_runs,val_WA_runs,test_UA_runs,test_WA_runs = KfoldCv(5, seed, config, label_files, sessions, max_text_tokens, max_audio_tokens, device, data_dir, models_dir)

# Calculate averages of metrics
train_UA_avg = np.mean(train_UA_runs)
train_WA_avg = np.mean(train_WA_runs)

val_UA_avg = np.mean(val_UA_runs)
val_WA_avg = np.mean(val_WA_runs)

test_UA_avg = np.mean(test_UA_runs)
test_WA_avg = np.mean(test_WA_runs)

# Report results
print("\nTraining: average UA: {}".format(train_UA_avg))
print("Training: average WA: {}".format(train_WA_avg))
print("\nValidation: average UA: {}".format(val_UA_avg))
print("Validation: average WA: {}".format(val_WA_avg))
print("\nTest: average UA: {}".format(test_UA_avg))
print("Test: average WA: {}".format(test_WA_avg))