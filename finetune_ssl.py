# Imports
import os
import pandas as pd
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
from test_function import test
from model_structure import newbob, Fusion, LHS
from utils import collate_tokens, collater, get_num_sentences, collate_batch

from fairseq.models.roberta import RobertaModel

# Define seed for reproducibility
seed = 2022
torch.manual_seed(seed)

def finetune(data_dir, models_dir, config, train_dataset, val_dataset, train_dataloader, val_dataloader, device):

    # Load sub-models
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt')

    # Freeze some layers
    speechBert_modules = [*speechBert.model.encoder.sentence_encoder.layers[:18]]
    Roberta_modules = [*roberta.model.encoder.sentence_encoder.layers[:9]]

    for module in speechBert_modules:
        for param in module.parameters():
            param.requires_grad = False

    for module in Roberta_modules:
        for param in module.parameters():
            param.requires_grad = False

    # Instantiate model class
    model = LHS(roberta, speechBert, freeze_models=config["freeze_models"]).to(device)

    # Instantiate optimizer class
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    # Instantiate learning rate scheduler class
    scheduler = config["scheduler"](optimizer, factor=config["factor"], model=model)

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
                                                                                                config["criterion"],
                                                                                                device,
                                                                                                step_train,
                                                                                                config["penalty"],
                                                                                                config["log_results"])
        step_train += curr_train_step

        # Validate model
        val_epoch_loss, val_epoch_u_accuracy, val_epoch_w_accuracy = validate(model, val_dataloader, val_dataset,
                                                                              config["criterion"], device, step_val, config["log_results"])
        step_val += 1


        print(f"Train Loss: {train_epoch_loss:.4f}, Train UA: {train_epoch_u_accuracy:.2f}, Train WA: {train_epoch_w_accuracy:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val UA: {val_epoch_u_accuracy:.2f}, Val WA: {val_epoch_w_accuracy:.2f}')

    """
    Save model
    """

    # serialize the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), data_dir + "outputs/fine_tuned_LHS_branch.pth")

    print('TRAINING COMPLETE')


