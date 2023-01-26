from typing import Any

import pandas as pd
import json


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    """Flatten one level of nesting"""
    return [item for sublist in list_of_lists for item in sublist]


def convert_char_to_word_pos(text: str, char_pos: int) -> int:
    """function which converts position based on character to position based on word"""
    return len(text[:char_pos].split())


def convert_word_to_char_pos(text: str, word_pos: int) -> int:
    """function which converts position based on word to position based on character"""
    return len(" ".join(text.split()[:word_pos]))


def convert_to_bio(text: str, entities=None) -> tuple[list[list[str]], list[list[str]]]:
    """Converts the text and entities to BIO format.
    Converts empty strings and [deleted by user] and [removed] into [UNK] token."""
    if entities is None:
        entities = []

    tokenized = [x.split() for x in text]
    bio_tokens = []

    for i in range(len(entities)):
        print(entities[i])
        print(text[i])

        if (
            text[i] in ["", "[deleted by user]\n[removed]"]
            or "[deleted]" in text[i]
            or "[removed]" in text[i]
            or "&#x200B;" in text[i]
        ):
            tokenized[i] = ["[UNK]"]
            bio_tokens.append(["O"])
            continue

        bio_tokens.append(["O" for _ in tokenized[i]])
        if entities[i]:
            for entity in entities[i]:
                start = entity["startOffset"]
                start = convert_char_to_word_pos(text[i], start)
                end = entity["endOffset"]
                end = convert_char_to_word_pos(text[i], end)

                label = entity["label"]
                bio_tokens[i][start] = "B-" + label
                for j in range(start + 1, end):
                    bio_tokens[i][j] = "I-" + label

            print(bio_tokens[i])
            print(tokenized[i])

    return tokenized, bio_tokens


if __name__ == "__main__":

    TASK = "test"

    df = pd.read_csv(f"../../data/external/st1_public_data/st1_{TASK}_inc_text.csv")

    try:
        entities = (
            df["stage1_labels"]
            .apply(lambda x: json.loads(x)[0]["crowd-entity-annotation"]["entities"])
            .tolist()
        )
    except KeyError:
        # for test data
        entities = []

    text = df["text"].tolist()

    tokenized, bio_tokens = convert_to_bio(text, entities)

    with open(f"../../data/interim/st1_{TASK}_bio_tokens.txt", "w") as f:
        for i in range(len(bio_tokens)):
            f.write(" ".join(bio_tokens[i]) + "\n")

    with open(f"../../data/interim/st1_{TASK}_tokens.txt", "w") as f:
        for i in range(len(tokenized)):
            f.write(" ".join(tokenized[i]) + "\n")
