import numpy as np
from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer
from transformers import DistilBertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
import torch
import re

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    id_x = 0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        id_x += 1
        print(id_x, doc_labels, doc_offset)
        if id_x == 32:
            print('stop')

        curr_enc = encodings[id_x-1]
        curr_text = texts[id_x-1]

        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        lower = (arr_offset[:, 0] == 0)
        upper = (arr_offset[:, 1] != 0)
        selected_mask = doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


class PICODataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


data_path = '../../data/interim'

with open(f"{data_path}/st1_train_tokens.txt", 'r') as f:
    texts = f.readlines()
texts = [x.split() for x in texts]

with open(f"{data_path}/st1_train_bio_tokens.txt", 'r') as f:
    tags = f.readlines()
tags = [x.split() for x in tags]
print("loaded")

# texts = texts[:20]
# tags = tags[:20]

# keep only first 400 tokens in tags list
tags = [x[:330] for x in tags]
texts = [x[:330] for x in texts]

train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2, random_state=42)

unique_tags = {tag for doc in tags for tag in doc}
tag2id = {tag: _id for _id, tag in enumerate(unique_tags)}
id2tag = {_id: tag for tag, _id in tag2id.items()}
print(tag2id)
print(id2tag)
print(unique_tags)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

for i in range(len(tags)):
    print(i, len(texts[i]), len(tags[i]), len(encodings['input_ids'][i]), len(encodings['offset_mapping'][i]))

train_labels = encode_tags(tags, encodings)


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = PICODataset(train_encodings, train_labels)
val_dataset = PICODataset(val_encodings, val_labels)

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()


# optimizer = AdamW(model.parameters(), lr=5e-6)
#
# dataloader = DataLoader(
#     dataset,
#     collate_fn=TraingingBatch,
#     batch_size=4,
#     shuffle=True,
# )
# for num, batch in enumerate(dataloader):
#     loss, logits = model(
#         input_ids=batch.input_ids,
#         attention_mask=batch.attention_masks,
#         labels=batch.labels,
#     )
#     loss.backward()
#     optimizer.step()
