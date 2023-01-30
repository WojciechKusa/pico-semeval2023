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

wandb.init(project="pico-semeval-task1")

data_path = "../../../data/interim"

with open(f"{data_path}/st1_train_tokens.txt", "r") as f:
    texts = f.readlines()
texts = [x.split() for x in texts]

# remove non-alphanumeric characters
texts = [[re.sub(r"\W+", "", x) for x in doc] for doc in texts]
# texts = [re.sub(r'\s', ' ', x) for x in texts]
# if token is empty string replace with UNK
texts = [[x if x != "" else "[UNK]" for x in doc] for doc in texts]

with open(f"{data_path}/st1_train_bio_tokens.txt", "r") as f:
    tags = f.readlines()
tags = [x.split() for x in tags]
print("loaded")

print(len(texts), len(tags))
# remove rows from tags and texts when text == [UNK]
tags = [y for x, y in zip(texts, tags) if x != ['UNK']]
texts = [x for x in texts if x != ['UNK']]
print(len(texts), len(tags))


texts = texts[:100]
tags = tags[:100]

# keep only first 400 tokens in tags list
tags = [x[:260] for x in tags]
texts = [x[:260] for x in texts]

train_texts, val_texts, train_tags, val_tags = train_test_split(
    texts, tags, test_size=0.2, random_state=42
)

label_list = list({tag for doc in tags for tag in doc})
tag2id = {tag: _id for _id, tag in enumerate(label_list)}
id2tag = {_id: tag for tag, _id in tag2id.items()}
print(tag2id)
print(id2tag)
print(label_list)

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
train_encodings = tokenizer(
    train_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)
val_encodings = tokenizer(
    val_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)

encodings = tokenizer(
    texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)


train_labels = encode_tags(train_tags, train_encodings, tag2id, texts)
val_labels = encode_tags(val_tags, val_encodings, tag2id, texts)

train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = PICODataset(train_encodings, train_labels)
val_dataset = PICODataset(val_encodings, val_labels)


results_folder = f"../../data/results/26jan/{model_name}"
# results_folder = f"/newstorage5/wkusa/models/pico/{model_name}"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

model = DistilBertForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list)
)
# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


training_args = TrainingArguments(
    output_dir=results_folder,
    num_train_epochs=2,
    evaluation_strategy = "epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    # warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../../../reports",
    logging_steps=10,
    report_to=["wandb"],
    full_determinism=True,
)

seqeval_metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    print(true_labels[0])
    print(true_predictions[0])
    print()
    print(true_labels[-1])
    print(true_predictions[-1])
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# save the model
trainer.save_model(results_folder)


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
test_predictions = trainer.predict(test_dataset)

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
