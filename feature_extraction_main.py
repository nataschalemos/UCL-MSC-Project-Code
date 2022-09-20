"""
This is the main script to create the final embeddings which will be the inputs to the model
"""

# Imports
import os
from feature_ext import create_bert_emb, EmotionDataPreprocessing, create_gpt2_tokens
from concatenate_embds import concatenate_tab_embs, concatenate_tab_embs_past

# Define working directory
wdir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'

# Change current working directory
os.chdir(wdir)
print("Current working directory: {0}".format(os.getcwd()))

"""
Create embeddings separately
"""
#
# # Create BERT embeddings
# create_bert_emb()
#
# # Create VQ-Wav2Vec tokens
# data_processor = EmotionDataPreprocessing()
# data_processor.preprocess_data(audio_path=wdir + 'Audio/', emb_path=wdir + 'vqw2vTokens/')
#
# # Create GPT-2 tokens
# create_gpt2_tokens()

"""
Create final TAB inputs ([-context,+context] context BERT embeddings)
"""

# # Concatenate context BERT embeddings for TAB input
# path_to_full_emb1 = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmb1/"
# concatenate_tab_embs(path_to_full_emb1, 1)
#
# path_to_full_emb2 = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmb2/"
# concatenate_tab_embs(path_to_full_emb2, 2)
#
# path_to_full_emb3 = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmb3/"
# concatenate_tab_embs(path_to_full_emb3, 3)

path_to_full_emb4 = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmb4/"
concatenate_tab_embs(path_to_full_emb4, 4)

# path_to_full_emb_past_2 = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmbPast2/"
# concatenate_tab_embs_past(path_to_full_emb_past_2, 2)