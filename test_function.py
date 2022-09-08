"""
Test function to test performance of trained model
"""

# Imports
import torch
from tqdm import tqdm
from torchmetrics import Accuracy

# validation function
def test(model, test_dataloader, test_dataset, criterion, device):

    print('Testing')

    model.eval()

    # define loss in each iteration
    test_running_loss = 0.0
    # define accuracy in each iteration
    # TODO: define weighted accuracy as well
    accuracy = Accuracy(num_classes=4, average='macro').to(device)
    test_running_acc = 0.0

    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size), position=0, leave=True)

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
            test_running_loss += loss.item()
            _, preds = torch.max(output_all.data, 1)

            # compute average class-wise accuracy
            test_running_acc += accuracy(preds.to(device), target.to(device)).item()

        # compute val loss after one epoch
        test_loss = test_running_loss / counter
        # compute val accuracy after one epoch
        test_accuracy = test_running_acc / counter

        return test_loss, test_accuracy