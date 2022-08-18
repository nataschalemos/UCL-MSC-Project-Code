"""
This is the main script to create the final embeddings which will be the inputs to the model
"""

# Imports
import os
from feature_ext import create_bert_emb, EmotionDataPreprocessing, create_gpt2_tokens
from concatenate_embds import concatenate_tab_embs

# Define working directory
wdir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'

# Change current working directory
os.chdir(wdir)
print("Current working directory: {0}".format(os.getcwd()))

"""
Create embeddings separately
"""

# Create BERT embeddings
create_bert_emb()

# Create VQ-Wav2Vec tokens
data_processor = EmotionDataPreprocessing()
data_processor.preprocess_data(audio_path=wdir + 'Audio/', emb_path=wdir + 'vqw2vTokens/')

# Create GPT-2 tokens
create_gpt2_tokens()

"""
Create final TAB inputs ([-3,+3]context BERT embeddings)
"""

# Concatenate context BERT embeddings for TAB input
concatenate_tab_embs()
