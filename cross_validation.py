# Imports
from sklearn.model_selection import KFold
from dataset_loader import IemocapDataset, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
import random

from utils import collate_tokens, collater, get_num_sentences, collate_batch
from model_structure import newbob, Fusion
from train_function import fit
from validation_function import validate
from test_function import test


def KfoldCv(fold, seed, config, label_files, sessions, max_text_tokens, max_audio_tokens, device, data_dir, roberta,
            speechBert):

    # Store results for each run
    train_UA_runs = np.zeros(fold)
    train_WA_runs = np.zeros(fold)
    val_UA_runs = np.zeros(fold)
    val_WA_runs = np.zeros(fold)
    test_UA_runs = np.zeros(fold)
    test_WA_runs = np.zeros(fold)

    # Define k-fold cross validation test harness
    kfold = KFold(n_splits=fold, shuffle=True, random_state=seed)

    # Loop over session folds
    for run, split_sess in tqdm(enumerate(kfold.split(sessions)), total=fold):

        train_label_files, val_label_files, test_label_files = collect_files(split_sess[0], split_sess[1], label_files, sessions)

        # Create train and validation dataloaders
        #train_dataloader, train_dataset = create_dataloader(split_sess[0], label_files, sessions, data_dir, config, device, max_text_tokens, max_audio_tokens)
        #val_dataloader, val_dataset = create_dataloader(split_sess[1], label_files, sessions, data_dir, config, device, max_text_tokens, max_audio_tokens, shuffle=False)

        train_dataloader, train_dataset = create_dataloader(train_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)
        val_dataloader, val_dataset = create_dataloader(val_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)
        test_dataloader, test_dataset = create_dataloader(test_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)

        # Instantiate model class
        model = Fusion(roberta, speechBert).to(device)

        # Instantiate optimizer class
        optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
        # Instantiate learning rate scheduler class
        scheduler = config["scheduler"](optimizer, factor=config["factor"], model=model)

        # Lists to store per-epoch loss and accuracy values
        train_loss, train_u_accuracy, train_w_accuracy = [], [], []
        val_loss, val_u_accuracy, val_w_accuracy = [], [], []

        # initialize step for WandB plots
        step_train = 0
        step_val = 0

        # Train the model
        for epoch in range(config["epochs"]):

            print(f'\nEpoch {epoch + 1} of {config["epochs"]}')

            # Fit model
            train_epoch_loss, train_epoch_u_accuracy, train_epoch_w_accuracy, curr_train_step = fit(model, train_dataloader, train_dataset,
                                                                          optimizer,
                                                                          config["criterion"], device, step_train)
            step_train += curr_train_step

            # Validate model
            val_epoch_loss, val_epoch_u_accuracy, val_epoch_w_accuracy = validate(model, val_dataloader, val_dataset, config["criterion"], device, step_val)
            step_val += 1

            train_loss.append(train_epoch_loss)
            train_u_accuracy.append(train_epoch_u_accuracy)
            train_w_accuracy.append(train_epoch_w_accuracy)
            val_loss.append(val_epoch_loss)
            val_u_accuracy.append(val_epoch_u_accuracy)
            val_w_accuracy.append(val_epoch_w_accuracy)

            print(f"Train Loss: {train_epoch_loss:.4f}, Train UA: {train_epoch_u_accuracy:.2f}, Train WA: {train_epoch_w_accuracy:.2f}")
            print(f'Val Loss: {val_epoch_loss:.4f}, Val UA: {val_epoch_u_accuracy:.2f}, Val WA: {val_epoch_w_accuracy:.2f}')

            # Update learning rate scheduler
            #scheduler.step(val_epoch_accuracy)

        """
        Save model
        """

        # # serialize the model to disk
        # print('Saving model...')
        # torch.save(model.state_dict(), data_dir + "outputs/model.pth")
        #
        # print('TRAINING COMPLETE')

        # Test model on test data
        test_loss, test_unweighted_accuracy, test_weighted_accuracy = test(model, test_dataloader, test_dataset, config["criterion"], device)

        # log metrics inside your val loop to visualize model performance
        wandb.run.summary({"test_unweighted_accuracy": test_unweighted_accuracy, "test_weighted_accuracy": test_weighted_accuracy, "test_loss": test_loss})

        print(f"Test Loss: {test_loss:.4f}, Test UA: {test_unweighted_accuracy:.2f}, Test WA: {test_weighted_accuracy:.2f}")

        # Save results
        train_UA_runs[run] = train_epoch_u_accuracy
        train_WA_runs[run] = train_epoch_w_accuracy

        val_UA_runs[run] = val_epoch_u_accuracy
        val_WA_runs[run] = val_epoch_w_accuracy

        test_UA_runs[run] = test_unweighted_accuracy
        test_WA_runs[run] = test_weighted_accuracy

    return train_UA_runs,train_WA_runs,val_UA_runs,val_WA_runs,test_UA_runs,test_WA_runs



def create_dataloader(label_files_set, data_dir, config, device, max_text_tokens, max_audio_tokens, shuffle=True):

    # # Convert session indexes into appropriate form
    # session = "|".join([sessions[s] for s in idxs])
    # # Collect set filenames
    # label_files_set = label_files[label_files[0].str.contains(session)]

    # Load dataset and dataloader
    dataset = IemocapDataset(labels_file=label_files_set,
                             dir=data_dir,
                             context_window=config["context_window"],
                             device=device,
                             max_text_tokens=int(max_text_tokens),
                             max_audio_tokens=int(max_audio_tokens),
                             transform=ToTensor())

    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                                  shuffle=shuffle, num_workers=0, collate_fn=collate_batch)

    return dataloader, dataset

def collect_files(train_sess, test_sess, label_files, sessions):

    # Convert session indexes into appropriate form
    train_sess = "|".join([sessions[s] for s in train_sess])
    test_sess = "|".join([sessions[s] for s in test_sess])

    train_label_files = label_files[label_files[0].str.contains(train_sess)]
    full_test_label_files = label_files[label_files[0].str.contains(test_sess)]

    id_list = ["_F", "_M"]
    random.shuffle(id_list)

    val_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[0])]
    test_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[1])]

    return train_label_files, val_label_files, test_label_files
