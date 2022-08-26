# Imports
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def collate_batch(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Roberta_tokens_list = [item['Roberta_tokens'] for item in batch]
    SpeechBERT_tokens_list = [item['SpeechBERT_tokens'] for item in batch]
    TAB_embedding = [item['TAB_embedding'] for item in batch]
    label = [item['label'] for item in batch]

    Roberta_tokens = pad_sequence(Roberta_tokens_list, padding_value=1, batch_first=True).to(device)
    SpeechBERT_tokens = pad_sequence(SpeechBERT_tokens_list, padding_value=1, batch_first=True).to(device)
    TAB_embedding = torch.stack(TAB_embedding).to(device)
    label = torch.stack(label).to(device)

    #return [Roberta_tokens, SpeechBERT_tokens, TAB_embedding, label]
    return {
        'SpeechBERT_tokens': SpeechBERT_tokens,
        'Roberta_tokens': Roberta_tokens,
        'TAB_embedding': TAB_embedding,
        'label': label
    }



def get_num_sentences(dataset_files):

    dataset_size = len(dataset_files)
    num_sentences = np.zeros(4, int)

    for idx in range(dataset_size):
        file_name, class_value = dataset_files.iloc[idx]
        if class_value == 1 or class_value == 2:
            num_sentences[1] += 1
        elif class_value == 0:
            num_sentences[class_value] += 1
        else:
            num_sentences[class_value-1] += 1

    return num_sentences

# def collate_tokens(values, pad_idx, max_target_value, eos_idx=None, left_pad=False,
#                    move_eos_to_beginning=False):
#     """Convert a list of 1d tensors into a padded 2d tensor."""
#
#     size = max_target_value  # max(v.size(0) for v in values) #Here the size can be fixed as 512
#     res = values[0].new(len(values), size).fill_(pad_idx)
#
#     def copy_tensor(src, dst):
#
#         if src.numel() > dst.numel():
#             clip_src = src[:dst.numel() - 1]
#             src = torch.cat((clip_src, torch.tensor([2])), 0)
#
#         assert dst.numel() == src.numel()
#         if move_eos_to_beginning:
#             assert src[-1] == eos_idx
#             dst[0] = eos_idx
#             dst[1:] = src[:-1]
#         else:
#             dst.copy_(src)
#
#     for i, v in enumerate(values):
#         copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
#     return res
#
# def collater(samples):
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     if len(samples) == 0:
#         return {}
#
#     ####################################################################
#     # collater for text chunks
#     #############################################
#     sources_text = [s['Roberta_tokens'] for s in samples]
#     sizes_text = [len(s) for s in sources_text]
#     max_target_size_t = min(max(sizes_text), 256)  # max text token seq length
#
#     collated_text = collate_tokens(sources_text, 1, max_target_size_t)  # 1 is the padding index
#
#     ####################################################################
#     # collater  for audio token chunks
#     #############################################
#     sources_audio_tokens = [s['SpeechBERT_tokens'] for s in samples]
#     sizes_audio = [len(s) for s in sources_audio_tokens]
#     max_target_size_a = min(max(sizes_audio), 1024)  # max audio token seq length
#
#     collated_audio_tokens = collate_tokens(sources_audio_tokens, 1,
#                                                       max_target_size_a)  # 1 is the padding index
#     TAB_embedding = [s['TAB_embedding'] for s in samples]
#     TAB_embedding = torch.stack(TAB_embedding)
#     #label = torch.LongTensor([s['label'] for s in samples])
#     label = [s['label'] for s in samples]
#     label = torch.stack(label)
#     #label = torch.FloatTensor([s['label'] for s in samples])
#
#     return {
#         'SpeechBERT_tokens': collated_audio_tokens.to(device),
#         'Roberta_tokens': collated_text.to(device),
#         'TAB_embedding': TAB_embedding.to(device),
#         # 'target': torch.LongTensor([int(s['target']) for s in samples])
#         'label': label.to(device)
#     }

def collate_tokens(values, pad_idx, max_target_value, eos_idx=None, left_pad=False,
                   move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    size = max_target_value  # max(v.size(0) for v in values) #Here the size can be fixed as 512
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):

        if src.numel() > dst.numel():
            clip_src = src[:dst.numel() - 1]
            src = torch.cat((clip_src, torch.tensor([2])), 0)

        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collater(samples):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(samples) == 0:
        return {}

    ####################################################################
    # collater for text chunks
    #############################################
    sources_text = [s['Roberta_tokens'] for s in samples]
    sizes_text = [len(s) for s in sources_text]
    max_target_size_t = min(max(sizes_text), 256)  # max text token seq length

    collated_text = collate_tokens(sources_text, 1, max_target_size_t)  # 1 is the padding index

    ####################################################################
    # collater  for audio token chunks
    #############################################
    sources_audio_tokens = [s['SpeechBERT_tokens'] for s in samples]
    sizes_audio = [len(s) for s in sources_audio_tokens]
    max_target_size_a = min(max(sizes_audio), 1024)  # max audio token seq length

    collated_audio_tokens = collate_tokens(sources_audio_tokens, 1,
                                                      max_target_size_a)  # 1 is the padding index
    TAB_embedding = [s['TAB_embedding'] for s in samples]
    TAB_embedding = torch.stack(TAB_embedding)
    #label = torch.LongTensor([s['label'] for s in samples])
    label = [s['label'] for s in samples]
    label = torch.stack(label)
    #label = torch.FloatTensor([s['label'] for s in samples])

    return {
        'SpeechBERT_tokens': collated_audio_tokens.to(device),
        'Roberta_tokens': collated_text.to(device),
        'TAB_embedding': TAB_embedding.to(device),
        # 'target': torch.LongTensor([int(s['target']) for s in samples])
        'label': label.to(device)
    }