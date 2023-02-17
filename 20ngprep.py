import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import tqdm


categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

df = pd.DataFrame(data_train.data)

words = {}

for doc_number in tqdm.tqdm(range(len(df))):
    for word in df.iloc[doc_number, 0].split():
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
    
words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}

# If word appears less than 10 times, remove it
for key in list(words.keys()):
    if words[key] < 20:
        del words[key]
    
# Remove words without letters
for key in list(words.keys()):
    if not any(char.isalpha() for char in key):
        del words[key]

# Write dictionary to file human readable
with open('words.txt', 'w') as f:
    for key, value in words.items():
        f.write('%s:%s' % (key, value))
        f.write('\n')