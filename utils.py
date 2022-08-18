# Imports
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Roberta_tokens_list = [item['Roberta_tokens'] for item in batch]
    SpeechBERT_tokens_list = [item['SpeechBERT_tokens'] for item in batch]
    TAB_embedding = [item['TAB_embedding'] for item in batch]
    label = [item['label'] for item in batch]

    Roberta_tokens = pad_sequence(Roberta_tokens_list, padding_value=0, batch_first=True).to(device)
    SpeechBERT_tokens = pad_sequence(SpeechBERT_tokens_list, padding_value=0, batch_first=True).to(device)
    TAB_embedding = torch.stack(TAB_embedding).to(device)
    label = torch.stack(label).to(device)

    return [Roberta_tokens, SpeechBERT_tokens, TAB_embedding, label]

