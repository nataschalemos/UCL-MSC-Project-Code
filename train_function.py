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

from torch.autograd import Variable


def fit(model, train_dataloader, train_dataset, optimizer, criterion, device, step):

    ##
    I = Variable(torch.zeros(16, 5, 5)) # bs, att_heads, att_heads
    for i in range(16):
        for j in range(5):
            I.data[i][j][j] = 1
    I = I.to(device)
    ##

    print('Training')

    # Log checkpoint
    batch_checkpoint = 20

    # define loss in each iteration
    train_running_loss = 0.0
    running_loss = 0.0

    # define accuracy in each iteration
    unweighted_accuracy = Accuracy(num_classes=4, average='macro').to(device)
    weighted_accuracy = Accuracy(num_classes=4, average='micro').to(device)
    train_running_u_acc = 0.0
    train_running_w_acc = 0.0
    running_u_acc = 0.0
    running_w_acc = 0.0

    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size), position=0, leave=True)

    # iterate through data batches
    for i, data in prog_bar:
    #for i, data in enumerate(train_dataloader):

        counter += 1

        Roberta_tokens, SpeechBERT_tokens, data_TAB, target = data['Roberta_tokens'], data['SpeechBERT_tokens'], data['TAB_embedding'], data['label']
        target = torch.argmax(torch.squeeze(target), dim=1)
        total += target.size(0)
        #optimizer.zero_grad()
        model.zero_grad()
        output_all, output_TAB = model(Roberta_tokens.to(device), SpeechBERT_tokens.to(device), data_TAB.to(device))

        # compute batch loss
        loss = criterion(output_all.to(device), target.to(device))

        #train_running_loss += loss.item() # replace current loss with penalized loss
        #running_loss += loss.item() # replace current loss with penalized loss

        ##
        # add penalization term
        attention = output_TAB.to(device)
        attentionT = torch.transpose(attention, 1, 2).contiguous()
        extra_loss = Frobenius(torch.bmm(attention, attentionT.to(device)) - I[:attention.size(0)])
        loss += 0.3 * extra_loss # penalization coef=0.3
        ##

        _, preds = torch.max(output_all.data, 1)

        # add number of correct predictions
        train_running_u_acc += unweighted_accuracy(preds.to(device), target.to(device)).item()
        train_running_w_acc += weighted_accuracy(preds.to(device), target.to(device)).item()

        # compute average class-wise accuracy
        running_u_acc += unweighted_accuracy(preds.to(device), target.to(device)).item()
        running_w_acc += weighted_accuracy(preds.to(device), target.to(device)).item()


        # calculate gradient
        loss.backward()
        # update model parameters
        optimizer.step()

        # Log metrics inside your training loop to visualize model performance
        # log every 20 mini-batches
        if i % batch_checkpoint == batch_checkpoint - 1:  # log every 20 mini-batches
            step += 1
            wandb.log({"train": {"loss": running_loss / batch_checkpoint,
                                 "unweighted_accuracy": running_u_acc / batch_checkpoint, "weighted_accuracy": running_w_acc / batch_checkpoint, "custom_step_train": step}})
            running_loss = 0.0
            running_u_acc = 0.0
            running_w_acc = 0.0

    # compute train loss after one epoch
    train_loss = train_running_loss / counter
    # compute train accuracy after one epoch
    train_unweighted_accuracy = train_running_u_acc / counter
    train_weighted_accuracy = train_running_w_acc / counter

    return train_loss, train_unweighted_accuracy, train_weighted_accuracy, step

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')
