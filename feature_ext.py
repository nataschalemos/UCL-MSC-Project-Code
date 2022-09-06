""""
Extract embeddings from IEMOCAP utterance files
- bert
- VQ-Wav2Vec speech tokens
- GPT-2 text tokens
"""

# Imports
import librosa
import os
import numpy as np
from tqdm import tqdm
import torch
import pickle

from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel


"""
Text Embedding Extraction
Adapted from: https://huggingface.co/bert-base-uncased
"""

def create_bert_emb():
    path_to_text = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/Text/"
    path_to_emb = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/BertEmb/"
    files = os.listdir(path_to_text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = BertModel.from_pretrained("bert-base-uncased")

    for f in files:
        if f.endswith(".txt"):
            txt_file = open(path_to_text + f, 'r')
            txt = txt_file.read()

            encoded_input = tokenizer(txt, return_tensors='pt')
            model.eval()
            output = model(**encoded_input)
            # Take the first vector of the hidden state, which is the token embedding of the classification [CLS] token.
            # Use this as the sentence embedding
            sent_emd = output.last_hidden_state[0][0]
            sent_emd_arr = sent_emd.detach().numpy()
            sent_emd_arr = np.expand_dims(sent_emd_arr, 0)

            np.savetxt(path_to_emb + f, sent_emd_arr, delimiter='\t')


"""
VQ-Wav2Vec Speech Tokens Extraction
Adapted from: https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/SPEECH-BERT-TOKENIZATION/convert_aud_to_token.py
"""

class EmotionDataPreprocessing():

    def __init__(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cp = torch.load('/Volumes/TOSHIBA EXT/Code/Models/vq-wav2vec_kmeans.pt', map_location=device)
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

        # Roberta wav2vec
        self.roberta = RobertaModel.from_pretrained('/Volumes/TOSHIBA EXT/Code/Models/bert_kmeans', checkpoint_file='bert_kmeans.pt')

        self.roberta.eval()

    def indices_to_string(self, idxs):
        # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
        return "<s>" + " " + " ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))

    def preprocess_data(self, audio_path, emb_path):

        num_items = 1e18
        current_num = 0

        # AUDIO
        if audio_path:
            audio_files = os.listdir(audio_path)
            print(len(audio_files), " audio_files found")
            prog_bar = tqdm(enumerate(audio_files), total=len(audio_files))

            for _, audio_file in prog_bar:

                if audio_file.endswith(".wav"):

                    audio_np, _ = librosa.load(audio_path + audio_file)
                    audio_features = torch.from_numpy(audio_np).unsqueeze(0)

                    # wav2vec
                    z = self.model.feature_extractor(audio_features)

                    _, idxs = self.model.vector_quantizer.forward_idx(z)

                    idx_str = self.indices_to_string(idxs)

                    tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True,
                                                                             add_if_not_exist=False).cpu().detach().numpy()

                    file_name = os.path.splitext(audio_file)[0]
                    output_file = emb_path + file_name + '.txt'

                    with open(output_file, 'w') as f:
                        for item in tokens:
                            f.write(str(item) + '\t')

                    current_num += 1

                    if current_num > num_items:
                        break

"""
GPT-2 Text Tokens Extraction
"""

# def create_gpt2_tokens():
#     path_to_text = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/Text/"
#     path_to_emb = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/GPT2Tokens/"
#     files = os.listdir(path_to_text)
#
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#
#     prog_bar = tqdm(enumerate(files), total=len(files))
#
#     for _, f in prog_bar:
#         if f.endswith(".txt"):
#             # Load audio transcriptions
#             txt_file = open(path_to_text + f, 'r')
#             txt = txt_file.read()
#
#             # Create tokens
#             tokens = tokenizer(txt)['input_ids']
#
#             # Save tokens
#             file_name = os.path.splitext(f)[0]
#             with open(path_to_emb + file_name + '.pkl', "wb") as fp:
#                 pickle.dump(tokens, fp)


def create_gpt2_tokens():
    path_to_text = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/Text/"
    path_to_emb = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/GPT2Tokens/"
    files = os.listdir(path_to_text)

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()

    prog_bar = tqdm(enumerate(files), total=len(files))

    for _, f in prog_bar:
        if f.endswith(".txt"):
            # Load audio transcriptions
            txt_file = open(path_to_text + f, 'r')
            txt = txt_file.read()

            # Create tokens
            tokens = roberta.encode(txt)
            tokens = tokens.tolist()

            # Save tokens
            output_file = path_to_emb + f

            with open(output_file, 'w') as f1:
                for item in tokens:
                    f1.write(str(item) + '\t')
