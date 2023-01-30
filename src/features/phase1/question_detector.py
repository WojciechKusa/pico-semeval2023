from typing import Union

import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix


def detect_question_span(text: str) -> list[dict[str, str]]:
    """Detects the question span in the text.
    Args:
        text (str): The text to detect the question span in.
    Returns:
        str: The question span.
    """
    questions = []
    question_marks = [m.start() for m in re.finditer(r"\?", text)]
    prev_question_end = 0

    for question_mark in question_marks:
        end = question_mark + 1
        if sentence_breakers := [
            m.start()
            for m in re.finditer(r"[\n/.?!]", text[prev_question_end:question_mark])
        ]:
            start = int(sentence_breakers[-1]) + prev_question_end + 1
        else:
            start = prev_question_end
        span = text[start:end]

        questions.append({"question_span": span, "start": start, "end": end})
        prev_question_end = end
    return questions


def convert_span_to_prediction_format(
    text: str, question_spans: list[dict[str, str]]
) -> np.ndarray:
    """Converts the question span to a prediction format.
    Args:
        text (str): The text to detect the question span in.
        question_spans (list[dict[str, str]]): The question spans.
    Returns:
        list[int]: The prediction format.
    """
    y_pred = np.zeros(len(text))
    if question_spans:
        for question in question_spans:
            y_pred[question["start"] : question["end"]] = 1
    return y_pred


def convert_from_char_to_word(y_pred: Union[np.ndarray, list[int]], text: str) -> np.ndarray:
    """Converts the prediction format from char to word level.
    Args:
        y_pred (list[int]): The prediction format.
        text (str): The text to detect the question span in.
    Returns:
        list[int]: The prediction format.
    """
    y_pred_word = np.zeros(len(text.split()))
    start_index = 0
    for i, word in enumerate(text.split()):
        if np.sum(y_pred[start_index : start_index + len(word)]) > 0:
            y_pred_word[i] = 1
        # y_pred = y_pred[len(word) :]
        start_index += len(word) + 1  # add space hence +1
    return y_pred_word


def evaluate_question_detector():
    task_id = "1"
    data_type = "train"
    input_file = f"../../../data/processed/st{task_id}_public_data/st{task_id}_{data_type}_parsed.tsv"
    df = pd.read_csv(input_file, encoding="utf-8", sep="\t")
    df = df[df["labels_char"] != "N.A."]
    y_true = df["labels_char"].tolist()
    y_true = [eval(x) for x in y_true]
    out_preds = []
    for text in tqdm(df["text"].tolist()):
        questions = detect_question_span(text)

        y_pred = convert_span_to_prediction_format(text, questions)

        out_preds.append(y_pred)

    result = sum(
        f1_score(y_true=first, y_pred=second, average="macro")
        for first, second in zip(y_true, out_preds)
    )
    print(result / len(y_true))

    y_true = [item for sublist in y_true for item in sublist]
    y_pred = [item for sublist in out_preds for item in sublist]

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    print(cm)
    cm = confusion_matrix(y_true, y_pred, normalize="pred")
    print(cm)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == "__main__":
    task_id = "1"
    data_type = "test"
    input_file = f"../../../data/interim/st1_{data_type}_tokens.txt"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    reference_file = f"../../../data/external/st1_public_data/st1_{data_type}.csv"
    df = pd.read_csv(reference_file, encoding="utf-8")

    out_preds = []
    for text in tqdm(lines):
        if text == "[UNK]":
            out_preds.append(np.zeros(1))
            continue

        questions = detect_question_span(text)
        y_pred = convert_span_to_prediction_format(text, questions)
        y_pred1 = convert_from_char_to_word(y_pred, text)

        out_preds.append(y_pred1)

    out_df = pd.DataFrame()
    for index_post, (row, pred) in tqdm(enumerate(zip(df.iterrows(), out_preds))):
        for index_word, p in enumerate(pred):
            out_df = pd.concat(
                [
                    out_df,
                    pd.DataFrame(
                        {
                            "text": row[1]["subreddit_id"],
                            "post_id": row[1]["post_id"],
                            "words": lines[index_post].split()[index_word],
                            "labels_char": p,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    out_df.to_csv(
        f"../../../data/processed/st1_{data_type}_predictions.csv",
        encoding="utf-8",
        index=False,
    )
