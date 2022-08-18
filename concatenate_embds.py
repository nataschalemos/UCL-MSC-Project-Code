"""
This script creates the final inputs to TAB by concatenating the embeddings appropriately.
"""

# Imports
import pandas as pd
import numpy as np


def concatenate_tab_embs():
    """
    Create concatenations for TAB input
    - concatenate context [-3, 3] BERT embeddings to create input vectors for tab with dims (1,768*7) for each utterance
    - use overlapping - shifting context window by u=1
    """

    root_dir = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/"
    path_to_embs = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/BertEmb/"
    path_to_full_emb = "/Volumes/TOSHIBA EXT/Code/IEMOCAP/FullBertEmb/"

    files = pd.read_csv(root_dir + "labels_train_t.txt", header=None, delimiter='\t', usecols=[0])

    # Separate filenames wrt sessions
    session1 = files[files[0].str.contains('s_1')].values.tolist()
    session2 = files[files[0].str.contains('s_2')].values.tolist()
    session3 = files[files[0].str.contains('s_3')].values.tolist()
    session4 = files[files[0].str.contains('s_4')].values.tolist()
    session5 = files[files[0].str.contains('s_5')].values.tolist()

    sessions = [session1, session2, session3, session4, session5]

    # Loop over session files
    for sess in sessions:
        # Compute total number of embeddings after grouping into 7 context embeddings (stays the same when using overlap)
        num_embds = len(sess)
        for i in range(num_embds):
            files_context_emb = np.zeros((7, 768))
            range_files = range(i-3,i+4)
            for num, idx in enumerate(range_files):
                if idx > 0 and idx < num_embds:
                    f = sess[idx]
                    file = f[0]
                    files_context_emb[num, :] = np.loadtxt(path_to_embs + file, delimiter='\t')

            files_context_emb_vec = files_context_emb.reshape(1, 768 * 7)
            np.savetxt(path_to_full_emb + sess[i][0], files_context_emb_vec, delimiter='\t')