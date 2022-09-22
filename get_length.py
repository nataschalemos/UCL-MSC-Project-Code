import os
import pandas as pd

label_files = pd.read_csv('/Volumes/TOSHIBA EXT/Code/IEMOCAP/labels_train_t.txt', header=None, delimiter='\t')

path = '/Volumes/TOSHIBA EXT/Code/IEMOCAP/vqw2vTokens/'

files_content = []
for filename in label_files[0]:
    filepath = os.path.join(path, filename)
    tokensized_text = []
    with open(filepath, mode='r') as f:
        for line in f:
            for word in line.strip().split('\t'):
                tokensized_text.append(int(word))
        files_content += [tokensized_text]

def FindMaxLength(lst):
    maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst)

    return maxList, maxLength

def Average(lst):
    i = [len(x) for x in lst]
    return sum(i) / len(i)

print(FindMaxLength(files_content))
print(Average(files_content))
