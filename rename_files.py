# This script is only used to rename the files of word and audio embeddings with respect to the original filenames
# Need to provide this script and the 'labels_train_t.txt' and 'labels_train_a.txt' files together!

# Imports
import pandas as pd
import sys
import os
import re

def rename_f(data_dir, emb_dir):

    labels_file = data_dir + 'labels_train_t.txt'
    label_files = pd.read_csv(labels_file, header=None, delimiter='\t')

    files = os.listdir(data_dir+emb_dir)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    i = 0

    for f in files:
        if f.startswith("._"):
            continue
        if f.endswith(".txt"):
            label_i = label_files[0][i]
            l_split = label_i.split('_')
            l_new = l_split[0] + "_" + l_split[2] + "_" + l_split[3] + "_" + l_split[4]
            assert f == l_new
            os.rename(data_dir+emb_dir+f, data_dir+emb_dir+label_i)
            i += 1

# Define path to directory with files
wdir = sys.argv[1]
# Define directory with data
data_dir = sys.argv[2]
# Define directory with embeddings (vqw2vTokens, GPT2Tokens, FullBertEmb1, FullBertEmb2 or FullBertEmb3)
emb_dir = sys.argv[3]


# wdir = '/Volumes/TOSHIBA EXT/Code/'
# data_dir = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/'
os.chdir(wdir)

# Rename files in directories
rename_f(data_dir, emb_dir + '/')

# rename_f(data_dir, 'vqw2vTokens/')
# rename_f(data_dir, 'GPT2Tokens/')
# rename_f(data_dir, 'FullBertEmb1/')
# rename_f(data_dir, 'FullBertEmb2/')
# rename_f(data_dir, 'FullBertEmb3/')
