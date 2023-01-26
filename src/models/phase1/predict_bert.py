import os.path

from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer
from transformers import DistilBertForTokenClassification
from sklearn.model_selection import train_test_split
import torch
import re
import numpy as np
import evaluate

import wandb


from utils import encode_tags, PICODataset

data_path = "../../../data/interim"

with open(f"{data_path}/st1_test_tokens.txt", "r") as f:
    texts = f.readlines()
texts = [x.split() for x in texts]

# remove non-alphanumeric characters
texts = [[re.sub(r"\W+", "", x) for x in doc] for doc in texts]
# texts = [re.sub(r'\s', ' ', x) for x in texts]
# if token is empty string replace with UNK
texts = [[x if x != "" else "[UNK]" for x in doc] for doc in texts]



with open(f"{data_path}/st1_test_bio_tokens.txt", "r") as f:
    tags = f.readlines()
tags = [x.split() for x in tags]
print("loaded")

print(len(texts), len(tags))
# remove rows from tags and texts when text == [UNK]
tags = [y for x, y in zip(texts, tags) if x != ['UNK']]
texts = [x for x in texts if x != ['UNK']]
print(len(texts), len(tags))


model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)


label_list = list({tag for doc in tags for tag in doc})
tag2id = {tag: _id for _id, tag in enumerate(label_list)}
id2tag = {_id: tag for tag, _id in tag2id.items()}
print(tag2id)
print(id2tag)
print(label_list)


path_to_model = f"../../data/results/{model_name}"
model = DistilBertForTokenClassification.from_pretrained(
    path_to_model,
    num_labels=len(label_list),
    id2label=id2tag,
    label2id=tag2id,
)


# encode test data and predict
test_encodings = tokenizer(
    texts[:100],
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)
test_encodings.pop("offset_mapping")
test_dataset = PICODataset(test_encodings, tags)
test_predictions = model.predict(test_dataset)

# get predictions
preds = np.argmax(test_predictions.predictions, axis=-1)
preds = [id2tag[x] for x in preds]

# get true labels
true = [id2tag[x] for x in test_predictions.label_ids]

# get test texts
test_texts = [tokenizer.convert_ids_to_tokens(x) for x in test_encodings["input_ids"]]
test_texts = [[x if x != "[UNK]" else "" for x in doc] for doc in test_texts]

# get test tags
test_tags = [tokenizer.convert_ids_to_tokens(x) for x in test_encodings["input_ids"]]
test_tags = [[x if x != "[UNK]" else "" for x in doc] for doc in test_tags]

# get test offsets
test_offsets = [tokenizer.decode(x, skip_special_tokens=True) for x in test_encodings["input_ids"]]
test_offsets = [[(x.start(), x.end()) for x in doc] for doc in test_offsets]

# print results
for i in range(len(test_texts)):
    print("Text:", test_texts[i])
    print("True:", true[i])
    print("Pred:", preds[i])
    print("Tags:", test_tags[i])
    print("Offsets:", test_offsets[i])
    print()
