import numpy as np
import torch


def encode_tags(tags, encodings, tag2id, texts):
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
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
