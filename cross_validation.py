# Imports
from sklearn.model_selection import KFold
from dataset_loader import IemocapDataset, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
import random
from fairseq.models.roberta import RobertaModel
import torch

from utils import collate_tokens, collater, get_num_sentences, collate_batch
from model_structure import newbob, Fusion
from train_function import fit
from validation_function import validate
from test_function import test


def KfoldCv(fold, seed, config, label_files, sessions, max_text_tokens, max_audio_tokens, device, data_dir, models_dir):

    # Store results for each run
    train_UA_runs = np.zeros(fold)
    train_WA_runs = np.zeros(fold)
    val_UA_runs = np.zeros(fold)
    val_WA_runs = np.zeros(fold)
    test_UA_runs = np.zeros(fold)
    test_WA_runs = np.zeros(fold)

    # Define k-fold cross validation test harness
    kfold = KFold(n_splits=fold, shuffle=True, random_state=seed)
    #kfold_splits = [split_sess for split_sess in kfold.split(sessions)]

    # Extend data split folds to include each possible val/test split of speakers in holdout session
    kfold_total_splits_1 = []
    kfold_total_splits_2 = []

    for split_sess in kfold.split(sessions):
        id_list = ["_F", "_M"]
        random.shuffle(id_list)
        kfold_total_splits_1.append(split_sess + tuple([id_list]))
        id_list.reverse()
        kfold_total_splits_2.append(split_sess + tuple([id_list]))

    kfold_total = kfold_total_splits_1 + kfold_total_splits_2

    print("\nStart 10-Fold Cross-Validation Procedure")

    group_id = wandb.util.generate_id()

    # Loop over session folds
    for run, split_sess in tqdm(enumerate(kfold_total), total=fold*2):

        # Start a W&B run
        # TODO: change from 3 to 4 before comitting code
        wandb_run = wandb.init(project="run-new-model-cv-4", group=group_id, name="run_"+str(run), entity="natascha-msc-project", config=config)
        # Save model inputs and hyperparameters
        wandb_run.config

        print("\nCV Run: {}".format(run+1))

        train_label_files, val_label_files, test_label_files = collect_files(split_sess[0], split_sess[1], split_sess[2], label_files, sessions)

        # Store number of utterances in each set for this cv split
        wandb.run.summary["train_utterances_" + str(run)] = len(train_label_files)
        wandb.run.summary["val_utterances_" + str(run)] = len(val_label_files)
        wandb.run.summary["test_utterances_" + str(run)] = len(test_label_files)
        # Store which session is the hold out session (for val + test split) and which speaker corresponds to the test set
        wandb.run.summary[str(run) + "_hold_out_sess_" + str(split_sess[1][0]) + "_test_speaker"] = split_sess[2][1].split('_')[1]

        # Create train and validation dataloaders
        #train_dataloader, train_dataset = create_dataloader(split_sess[0], label_files, sessions, data_dir, config, device, max_text_tokens, max_audio_tokens)
        #val_dataloader, val_dataset = create_dataloader(split_sess[1], label_files, sessions, data_dir, config, device, max_text_tokens, max_audio_tokens, shuffle=False)

        train_dataloader, train_dataset = create_dataloader(train_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)
        val_dataloader, val_dataset = create_dataloader(val_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)
        test_dataloader, test_dataset = create_dataloader(test_label_files, data_dir, config,
                                                            device, max_text_tokens, max_audio_tokens)

        # Load sub-models
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').to(device)
        speechBert = RobertaModel.from_pretrained(models_dir, checkpoint_file='bert_kmeans.pt').to(device)

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

        #last_loss = 1000
        #trigger_times = 0

        # initialize step for WandB plots
        step_train = 0
        step_val = 0

        # Train the model
        for epoch in range(config["epochs"]):

            print(f'\nEpoch {epoch + 1} of {config["epochs"]}')

            # Fit model
            model.train()
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

            # Early stopping
            if val_epoch_loss > last_best_loss:
                trigger_times_best += 1
                if trigger_times_best >= config["patience"]:
                    print('\nEarly stopping! \n loss has not improved from lowest loss.')
                    break
            else:
                trigger_times_best = 0
                last_best_loss = val_epoch_loss

            # if val_epoch_loss > last_loss:
            #     trigger_times += 1
            #     if trigger_times >= patience:
            #         print('\nEarly stopping! \n loss no longer decreasing.')
            #         break
            # else:
            #     trigger_times = 0
            #
            # last_loss = val_epoch_loss

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
        test_loss, test_unweighted_accuracy, test_weighted_accuracy, test_class_accuracy = test(model, test_dataloader, test_dataset, config["criterion"], device)

        # log metrics inside your val loop to visualize model performance
        wandb.run.summary["test_unweighted_accuracy_" + str(run)] = test_unweighted_accuracy
        wandb.run.summary["test_weighted_accuracy_" + str(run)] = test_weighted_accuracy
        wandb.run.summary["test_class_0_accuracy_" + str(run)] = test_class_accuracy[0]
        wandb.run.summary["test_class_1_accuracy_" + str(run)] = test_class_accuracy[1]
        wandb.run.summary["test_class_2_accuracy_" + str(run)] = test_class_accuracy[2]
        wandb.run.summary["test_class_3_accuracy_" + str(run)] = test_class_accuracy[3]
        wandb.run.summary["test_loss_" + str(run)] = test_loss

        print(f"Test Loss: {test_loss:.4f}, Test UA: {test_unweighted_accuracy:.2f}, Test WA: {test_weighted_accuracy:.2f}")

        # Save results
        train_UA_runs[run] = train_epoch_u_accuracy
        train_WA_runs[run] = train_epoch_w_accuracy

        val_UA_runs[run] = val_epoch_u_accuracy
        val_WA_runs[run] = val_epoch_w_accuracy

        test_UA_runs[run] = test_unweighted_accuracy
        test_WA_runs[run] = test_weighted_accuracy

        wandb_run.finish()

    return train_UA_runs, train_WA_runs, val_UA_runs, val_WA_runs, test_UA_runs, test_WA_runs



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

# def collect_files(train_sess, test_sess, label_files, sessions):
#
#     # Convert session indexes into appropriate form
#     train_sess = "|".join([sessions[s] for s in train_sess])
#     test_sess = "|".join([sessions[s] for s in test_sess])
#
#     train_label_files = label_files[label_files[0].str.contains(train_sess)]
#     full_test_label_files = label_files[label_files[0].str.contains(test_sess)]
#
#     id_list = ["_F", "_M"]
#     random.shuffle(id_list)
#
#     val_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[0])]
#     test_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[1])]
#
#     return train_label_files, val_label_files, test_label_files

def collect_files(train_sess, test_sess, id_list, label_files, sessions):

    # Convert session indexes into appropriate form
    train_sess = "|".join([sessions[s] for s in train_sess])
    test_sess = "|".join([sessions[s] for s in test_sess])

    train_label_files = label_files[label_files[0].str.contains(train_sess)]
    full_test_label_files = label_files[label_files[0].str.contains(test_sess)]

    val_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[0])]
    test_label_files = full_test_label_files[full_test_label_files[0].str.contains(id_list[1])]

    return train_label_files, val_label_files, test_label_files

