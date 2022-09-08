"""
Validation function for each epoch
Adapted from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""

# Imports
import torch
from tqdm import tqdm
import wandb
from torchmetrics import Accuracy

# validation function
def validate(model, test_dataloader, val_dataset, criterion, device, step):

    print('Validating')

    model.eval()

    # define loss in each iteration
    val_running_loss = 0.0
    # define accuracy in each iteration
    unweighted_accuracy = Accuracy(num_classes=4, average='macro').to(device)
    weighted_accuracy = Accuracy(num_classes=4, average='micro').to(device)
    val_running_u_acc = 0.0
    val_running_w_acc = 0.0

    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset) / test_dataloader.batch_size), position=0, leave=True)

    with torch.no_grad():

        for i, data in prog_bar:

            counter += 1
            #Roberta_tokens, SpeechBERT_tokens, data_TAB, target = data[0], data[1], data[2], data[3]
            Roberta_tokens, SpeechBERT_tokens, data_TAB, target = data['Roberta_tokens'], data['SpeechBERT_tokens'], \
                                                                  data['TAB_embedding'], data['label']
            target = torch.argmax(torch.squeeze(target), dim=1)
            total += target.size(0)
            output_all, output_TAB = model(Roberta_tokens.to(device), SpeechBERT_tokens.to(device), data_TAB.to(device))

            # compute batch loss
            loss = criterion(output_all.to(device), target.to(device))
            val_running_loss += loss.item()
            _, preds = torch.max(output_all.data, 1)

            # compute average class-wise accuracy
            val_running_u_acc += unweighted_accuracy(preds.to(device), target.to(device)).item()
            val_running_w_acc += weighted_accuracy(preds.to(device), target.to(device)).item()

        # compute val loss after one epoch
        val_loss = val_running_loss / counter
        # compute val accuracy after one epoch
        val_unweighted_accuracy = val_running_u_acc / counter
        val_weighted_accuracy = val_running_w_acc / counter

        # log metrics inside your val loop to visualize model performance
        wandb.log({"val": {"loss": val_loss, "unweighted_accuracy": val_unweighted_accuracy, "weighted_accuracy": val_weighted_accuracy, "custom_step_val": step}})

        return val_loss, val_unweighted_accuracy, val_weighted_accuracy