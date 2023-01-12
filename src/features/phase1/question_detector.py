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


if __name__ == "__main__":
    task_id = "1"
    input_file = (
        f"../../../data/processed/st{task_id}_public_data/st{task_id}_train_parsed.tsv"
    )
    df = pd.read_csv(input_file, encoding="utf-8", sep="\t")
    df = df[df["labels_char"] != "N.A."]
    y_true = df["labels_char"].tolist()
    y_true = [eval(x) for x in y_true]
    out_preds = []
    for text in tqdm(df["text"].tolist()):
        questions = detect_question_span(text)

        y_pred = np.zeros(len(text))
        if questions:
            for question in questions:
                y_pred[question["start"] : question["end"]] = 1

        out_preds.append(y_pred)

    # df['question_pred'] = out_preds

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
