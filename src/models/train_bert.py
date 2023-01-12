import os.path

from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer
from transformers import DistilBertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
import torch
import re
import numpy as np
import evaluate
from datasets import load_metric

import wandb

wandb.init(project="pico-semeval-task1")



def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for id_x, (doc_labels, doc_offset) in enumerate(
        zip(labels, encodings.offset_mapping)
    ):
        # print(id_x, doc_labels, doc_offset)
        if id_x == 32:
            print("stop")

        curr_enc = encodings[id_x]
        curr_text = texts[id_x]
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        lower = arr_offset[:, 0] == 0
        upper = arr_offset[:, 1] != 0
        selected_mask = doc_enc_labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        ]
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


class PICODataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


data_path = "../../data/interim"

with open(f"{data_path}/st1_train_tokens.txt", "r") as f:
    texts = f.readlines()
texts = [x.split() for x in texts]

# remove non alphanumeric characters
texts = [[re.sub(r"\W+", "", x) for x in doc] for doc in texts]
# texts = [re.sub(r'\s', ' ', x) for x in texts]
# if token is empty string replace with UNK
texts = [[x if x != "" else "[UNK]" for x in doc] for doc in texts]

with open(f"{data_path}/st1_train_bio_tokens.txt", "r") as f:
    tags = f.readlines()
tags = [x.split() for x in tags]
print("loaded")

# texts = texts[:500]
# tags = tags[:500]

# keep only first 400 tokens in tags list
tags = [x[:260] for x in tags]
texts = [x[:260] for x in texts]

train_texts, val_texts, train_tags, val_tags = train_test_split(
    texts, tags, test_size=0.2, random_state=42
)

unique_tags = {tag for doc in tags for tag in doc}
tag2id = {tag: _id for _id, tag in enumerate(unique_tags)}
id2tag = {_id: tag for tag, _id in tag2id.items()}
print(tag2id)
print(id2tag)
print(unique_tags)
label_list = list(unique_tags)

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


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = PICODataset(train_encodings, train_labels)
val_dataset = PICODataset(val_encodings, val_labels)

results_folder = f"../../data/results/{model_name}"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

model = DistilBertForTokenClassification.from_pretrained(
    model_name, num_labels=len(unique_tags)
)
# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


training_args = TrainingArguments(
    output_dir=results_folder,
    num_train_epochs=10,
    evaluation_strategy = "epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    # warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../../reports",
    logging_steps=10,
    # use_mps_device=True,
    report_to=["wandb"],
    full_determinism=True,
)


acc = evaluate.load("accuracy")
macro_f1 = evaluate.load("f1")
seqeval_metric = load_metric("seqeval")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     acc_score = 0
#     f1_score = 0
#     for pred, true, in zip(predictions, labels):
#         acc_score += acc.compute(predictions=pred, references=true)['accuracy']
#         f1_score += macro_f1.compute(predictions=pred, references=true, average='macro')['f1']
#
#     return {
#         "accuracy": acc_score/len(predictions),
#         "f1": f1_score/len(predictions)
#     }


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
    print(true_labels[0], true_predictions[0])
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# train model on gpu


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
# trainer.add_callback(CustomCallback(trainer))

trainer.train()

trainer.evaluate()

# # encode test data and predict
# test_encodings = tokenizer(
#     texts[:100],
#     is_split_into_words=True,
#     return_offsets_mapping=True,
#     padding=True,
#     truncation=True,
# )
# test_encodings.pop("offset_mapping")
# test_dataset = PICODataset(test_encodings, tags)
# test_predictions = trainer.predict(test_dataset)
#
# # get predictions
# preds = np.argmax(test_predictions.predictions, axis=-1)
# preds = [id2tag[x] for x in preds]
#
# # get true labels
# true = [id2tag[x] for x in test_predictions.label_ids]
#
# # get test texts
# test_texts = [tokenizer.convert_ids_to_tokens(x) for x in test_encodings["input_ids"]]
# test_texts = [[x if x != "[UNK]" else "" for x in doc] for doc in test_texts]
#
# # get test tags
# test_tags = [tokenizer.convert_ids_to_tokens(x) for x in test_encodings["input_ids"]]
# test_tags = [[x if x != "[UNK]" else "" for x in doc] for doc in test_tags]
#
# # get test offsets
# test_offsets = [tokenizer.decode(x, skip_special_tokens=True) for x in test_encodings["input_ids"]]
# test_offsets = [[(x.start(), x.end()) for x in doc] for doc in test_offsets]
#
# # print results
# for i in range(len(test_texts)):
#     print("Text:", test_texts[i])
#     print("True:", true[i])
#     print("Pred:", preds[i])
#     print("Tags:", test_tags[i])
#     print("Offsets:", test_offsets[i])
#     print()
