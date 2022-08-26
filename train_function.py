"""
Train function for each epoch
Adapted from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""

# Imports
import torch
from torchmetrics import Accuracy
from tqdm import tqdm
import wandb
import numpy as np


def fit(model, train_dataloader, train_dataset, optimizer, criterion, device, step):

    print('Training')

    # Log checkpoint
    batch_checkpoint = 20

    # define loss in each iteration
    train_running_loss = 0.0
    running_loss = 0.0

    # define accuracy in each iteration
    accuracy = Accuracy(num_classes=4, average='macro').to(device)
    train_running_correct = 0.0
    running_acc = 0.0

    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))

    # iterate through data batches
    for i, data in prog_bar:
    #for i, data in enumerate(train_dataloader):

        counter += 1

        #Roberta_tokens, SpeechBERT_tokens, data_TAB, target = data[0], data[1], data[2], data[3]
        Roberta_tokens, SpeechBERT_tokens, data_TAB, target = data['Roberta_tokens'], data['SpeechBERT_tokens'], data['TAB_embedding'], data['label']
        target = torch.argmax(torch.squeeze(target), dim=1)
        total += target.size(0)
        optimizer.zero_grad()
        output_all, output_TAB = model(Roberta_tokens.to(device), SpeechBERT_tokens.to(device), data_TAB.to(device))

        # compute batch loss
        loss = criterion(output_all.to(device), target.to(device))
        train_running_loss += loss.item()
        running_loss += loss.item()
        _, preds = torch.max(output_all.data, 1)

        # add number of correct predictions
        train_running_correct += accuracy(preds.to(device), target.to(device)).item()

        # compute average class-wise accuracy
        running_acc += accuracy(preds.to(device), target.to(device)).item()

        # calculate gradient
        loss.backward()
        # update model parameters
        optimizer.step()

        # Log metrics inside your training loop to visualize model performance
        # log every 20 mini-batches
        if i % batch_checkpoint == batch_checkpoint - 1:  # log every 10 mini-batches
            step += 1
            wandb.log({"train": {"loss": running_loss / batch_checkpoint,
                                 "accuracy": running_acc / batch_checkpoint, "custom_step_train": step}})
            running_loss = 0.0
            running_acc = 0.0

    # compute train loss after one epoch
    train_loss = train_running_loss / counter
    # compute train accuracy after one epoch
    train_accuracy = train_running_correct / counter

    return train_loss, train_accuracy, step