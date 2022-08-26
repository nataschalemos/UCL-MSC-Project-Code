# Multimodal Emotion Recognition

This project implements a model to perform multimodal emotion recognition from text and audio.

Parts of two models proposed in papers [1,2] were merged together to make up the final model architecture used in this implementation.
Namely, the Time-Asynchronous Branch (TAB) from [1] and the full model architecture in [2] were merged.
The outputs of the 3 sub-models were fused using a simple concatenation technique and this was then passed to a 
classification head which produced the final output of the model. 


## Implementation

#### 1) Creating dataset
The IEMOCAP dataset is used [3, 4] which contains a total of 5 independent sessions each corresponding to a dyadic conversation between a female and a male speaker. In each session there are multiple audio files which correspond to a running dialogue with the corresponding transcriptions saved as text files. 

The standard baseline for training and evaluation across different studies in multi-modal emotion recognition using IEMOCAP dataset is done using only 5 emotions: angry, happy, excited, sad and neutral, where happy and excited are merged. Therefore, in the method described below for processing the dataset used in this implementation, only utterances corresponding to these 5 emotions were kept and the rest were discarded.

From the IEMOCAP dataset, utterance-level audio (.wav) files and corresponding transcripts (.txt) were extracted and saved under filename “index_s_session#_emotion.wav” and “index_s_session#_emotion.txt” (e.g. “0_s_1_neu.wav” and “0_s_1_neu.txt”) in separate directories named Audio and Text, respectively.

Utterance-level text and audio files are created by running mocap_data_collect.py.

#### 2) Creating Embeddings
Embeddings are created individually for each utterance file and saved under the same filename as the one corresponding to the original utterance.
######The code implementation for creating the utterance-level files can be found in: feature_ext.py.

A) BERT Text Embeddings
Sentence embeddings are created using BERT for each utterance individually. These 768-d BERT embeddings can be found in the directory called BertEmb.

B) GPT-2 Text Tokens
Text tokens were generated for each utterance using the GPT-2 tokenizer. These can be found in the directory called GPT2Tokens.

C) VQ-Wav2Vec Audio Tokens
Audio tokens were generated for each utterance using a pre-trained VQ-Wav2Vec model. These can be found in the directory called vqw2vTokens.

#### 3) Pre-processing Embeddings
Using the individual BERT Text embeddings created in the previous step, embeddings for a number of consecutive utterances were concatenated to create context BERT embeddings.

A context of [-3, +3] was used (which corresponds to the three utterances before and after the utterance at the current time step) and hence the total size of the context BERT embeddings vector is 768*7=5376. Also, overlapping of the context window was used (with a shift of 1 utterance), which means that a context vector was created for each utterance in the dataset. For the first three and last three sentence embeddings, 768-d zero-vectors were used for padding.

The final context embeddings can be found in the directory called FullBertEmb.

The code implementation can be found in: concatenate_embds.py.
All final embeddings are created by running feature_extraction_main.py.

#### 3) Training Model
The model architecture can be found in model_structure.py
The custom dataset loader can be found in dataset_loader.py
The training loop for one epoch is implemented in train_function.py
The validation loop for one epoch is implemented in validation_function.py
The learning scheduler used for training is implemented in model_structure.py

The model is trained using a custom dataset loader where the embeddings for each utterance are loaded and processed. This involves truncating token vectors to a max length. Since there are different sized audio and text tokens, the dataloader used performs zero padding to make all token vectors in a batch equally sized.

## Running instructions

The packages required to run this code are presented in requirements.txt.

Additionally, two models need to be downloaded using the following links:

1. VQ-wav2vec - https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
2. Roberta - https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar

To train the model run the following command in the terminal:

python main.py <path_to_dir> <path_to_data> <path_to_models>

- <path_to_dir>: path to directory with scripts
- <path_to_data>: path to directory with data
- <path_to_models> path to directory containing downloaded models
- <context_window> context value to create BERT sentence context embeddings (can be 1, 2 or 3)
- <max_text_tokens> maximum sequence length for text tokens
- <max_audio_tokens> maximum sequence length for audio tokens

## References
[1] https://arxiv.org/pdf/2010.14102.pdf

[2] https://arxiv.org/pdf/2008.06682.pdf






