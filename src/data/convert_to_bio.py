import pandas as pd
import numpy as np
import json
import re


train_df = pd.read_csv('../../data/external/st1_public_data/st1_train_inc_text.csv')
test_df = pd.read_csv('../../data/external/st1_public_data/st1_test_inc_text.csv')

entities = train_df['stage1_labels'].apply(lambda x: json.loads(x)[0]['crowd-entity-annotation']['entities']).tolist()
# train_df['len_stage1_labels'] = train_df['stage1_labels'].apply(lambda x: len(x))

text = train_df['text'].tolist()

# re.sub(r'\s', ' ', text[0])


# function which converts position based on character to position based on word
def convert_char_to_word_pos(text, char_pos):
    return len(text[:char_pos].split())


# function which converts position based on word to position based on character
def convert_word_to_char_pos(text, word_pos):
    return len(' '.join(text.split()[:word_pos]))


tokenized = [x.split() for x in text]
bio_tokens = []

for i in range(len(entities)):
    print(entities[i])
    print(text[i])

    if text[i] in ['', '[deleted by user]\n[removed]'] or '[deleted]' in text[i] or '[removed]' in text[i] or '&#x200B;' in text[i]:
        tokenized[i] = ['[UNK]']
        bio_tokens.append(['O'])
        # bio_tokens.append(['O'] * len(tokenized[i]))
        continue

    if not entities[i]:
        bio_tokens.append(['O' for x in tokenized[i]])
        # continue
    else:
        bio_tokens.append(['O' for x in tokenized[i]])
        for entity in entities[i]:
            start = entity['startOffset']
            start = convert_char_to_word_pos(text[i], start)
            end = entity['endOffset']
            end = convert_char_to_word_pos(text[i], end)

            label = entity['label']
            bio_tokens[i][start] = 'B-' + label
            for j in range(start+1, end):
                bio_tokens[i][j] = 'I-' + label

        # # print(text[i])
        # for entity in entities[i]:
        #     print(entity['startOffset'], entity['endOffset'], entity['label'])
        #     print(text[i][entity['startOffset']:entity['endOffset']])
        #

        print(bio_tokens[i])
        print(tokenized[i])


# flattens two lists
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# df = pd.DataFrame(data=[flatten(tokenized), flatten(bio_tokens)], index=['Tokens', 'Tags'])
# df.transpose().to_csv('train.csv')

with open('../../data/interim/st1_train_bio_tokens.txt', 'w') as f:
    for i in range(len(bio_tokens)):
        f.write(' '.join(bio_tokens[i]) + '\n')

with open('../../data/interim/st1_train_tokens.txt', 'w') as f:
    for i in range(len(tokenized)):
        f.write(' '.join(tokenized[i]) + '\n')
