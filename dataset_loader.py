# Custom dataset to wrap TSB and TAB datasets

"""
Code adapted and modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import fairseq
import librosa
# from torchvision import transforms
import pickle


class IemocapDataset(Dataset):
    """IEMOCAP dataset"""

    def __init__(self, labels_file, dir, max_text_tokens=512, max_audio_tokens=512, transform=None):
        """
        Args:
            labels_file (string): Path to the csv file with labels. ('/Volumes/TOSHIBA EXT/Code/IEMOCAP/labels_train_t.txt')
            root_dir (string): Directory with embeddings. ('/Volumes/TOSHIBA EXT/Code/IEMOCAP')
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels_file
        self.dir = dir
        self.max_text_tokens = max_text_tokens
        self.max_audio_tokens = max_audio_tokens
        self.transform = transform

    def __len__(self):
        # Returns the size of the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract indexed sample filename and label
        file_name, class_value = self.labels.iloc[idx]
        file_name = os.path.splitext(file_name)[0]

        SpeechBERT_tks_file_name = os.path.join(self.dir + 'vqw2vTokens', file_name + '.txt')
        Roberta_tks_file_name = os.path.join(self.dir + 'GPT2Tokens', file_name + '.pkl')
        TAB_emb_file_name = os.path.join(self.dir + 'FullBertEmb', file_name + '.txt')

        # Load sample embedding vectors
        open_file = open(Roberta_tks_file_name, "rb")
        Roberta_tokens = pickle.load(open_file)
        open_file.close()
        if len(Roberta_tokens) > self.max_text_tokens:
            Roberta_tokens = Roberta_tokens[:self.max_text_tokens]

        SpeechBERT_tokens = np.loadtxt(SpeechBERT_tks_file_name)
        if SpeechBERT_tokens.size > self.max_audio_tokens:
            SpeechBERT_tokens = SpeechBERT_tokens[:self.max_audio_tokens]

        TAB_embedding = np.loadtxt(TAB_emb_file_name)

        # One-hot encode label (EMOTION_ENCODING = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'neu': 4})
        # merge "happy" and "excited" labels
        label = np.zeros((1, 4))
        if class_value == 1 or class_value == 2:
            label[0, 1] = 1
        elif class_value == 0:
            label[0, class_value] = 1
        else:
            label[0, class_value - 1] = 1

        sample = {'Roberta_tokens': Roberta_tokens, 'SpeechBERT_tokens': SpeechBERT_tokens,
                  'TAB_embedding': TAB_embedding, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        Roberta_tokens, SpeechBERT_tokens, TAB_embedding, label = sample['Roberta_tokens'], sample['SpeechBERT_tokens'], \
                                                                  sample['TAB_embedding'], sample['label']

        Roberta_tokens = torch.tensor(Roberta_tokens)
        Roberta_tokens = Roberta_tokens.long()
        SpeechBERT_tokens = torch.from_numpy(SpeechBERT_tokens)
        SpeechBERT_tokens = SpeechBERT_tokens.long()
        TAB_embedding = torch.from_numpy(TAB_embedding)
        label = torch.from_numpy(label)

        return {'Roberta_tokens': Roberta_tokens,
                'SpeechBERT_tokens': SpeechBERT_tokens,
                'TAB_embedding': TAB_embedding.float(),
                'label': label.float()}
