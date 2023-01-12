import pandas as pd
import spacy
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

nlp = spacy.load("en_core_sci_sm")


def detect_question_span(text: str) -> list[dict[str, str]]:
    """Detects the question span in the text.
    Args:
        text (str): The text to detect the question span in.
    Returns:
        str: The question span.
    """
    questions = []
    # find all positions of question marks
    question_marks = [m.start() for m in re.finditer(r"\?", text)]
    prev_question_end = 0

    for question_mark in question_marks:
        # find the start character of a sentence before the question mark
        end = question_mark + 1
        if sentence_breakers := [
            m.start()
            for m in re.finditer(r"[\n/.?]", text[prev_question_end:question_mark])
        ]:
            start = int(sentence_breakers[-1]) + prev_question_end + 1
        else:
            start = prev_question_end
        # start = text.rfind("[\n.?]", prev_question_end, question_mark) + 1
        span = text[start:end]

        questions.append({"question_span": span, "start": start, "end": end})
        prev_question_end = end
    return questions


def detect_question_span2(text: str) -> list[dict[str, str]]:
    questions = []
    doc = nlp(text)

    for sent in doc.sents:
        if sent.text.endswith("?"):
            question_span = sent.text
            # get start and end char position of question span
            start = sent.start_char
            end = sent.end_char
            questions.append(
                {"question_span": question_span, "start": start, "end": end}
            )
            break

    return questions


def detect_question_span3(text: str) -> list[dict[str, str]]:
    questions = []
    # find all positions of question marks
    question_marks = [m.start() for m in re.finditer(r"\?", text)]
    if not question_marks:
        return questions
    prev_question_end = 0
    if text[0] == "?":
        question_marks.insert(0, 0)
    for question_mark in question_marks:
        # find the start character of a sentence before the question mark
        end = question_mark
        start = text.rfind("[\n.?]", prev_question_end, question_mark) + 1
        span = text[start:end]
        questions.append({"question_span": span, "start": start, "end": end})
        prev_question_end = end - 1
    return questions


if __name__ == "__main__":
    task_id = "1"
    input_file = (
        f"../../../data/processed/st{task_id}_public_data/st{task_id}_train_parsed.tsv"
    )
    df = pd.read_csv(input_file, encoding="utf-8", sep="\t")

    df = df[df["labels_char"] != "N.A."]
    first_n = 300
    y_true = df["labels_char"].tolist()[:first_n]

    y_true = [eval(x) for x in y_true]
    # print(len(df['labels_char'].tolist()[0]))
    out_preds = []
    for text in tqdm(df["text"].tolist()[:first_n]):
        questions = detect_question_span(text)
        # questions2 = detect_question_span2(text)
        # questions3 = detect_question_span3(text)
        # print(f"{questions=}")
        # print(f"{questions2=}")
        # print(f"{questions3=}")

        y_pred = np.zeros(len(text))
        if questions:
            for question in questions:
                # print(question['question_span'], question['start'], question['end'])
                y_pred[question["start"] : question["end"]] = 1

        out_preds.append(y_pred)

    # df['y_pred'] = out_preds

    result = 0
    for first, second in zip(y_true, out_preds):
        result += f1_score(y_true=first, y_pred=second, average="macro")

    print(result / len(y_true))

    # flatten list y_true
    y_true = [item for sublist in y_true for item in sublist]
    # flatten list y_pred
    y_pred = [item for sublist in out_preds for item in sublist]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    print(cm)

    cm = confusion_matrix(y_true, y_pred, normalize="pred")
    print(cm)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
