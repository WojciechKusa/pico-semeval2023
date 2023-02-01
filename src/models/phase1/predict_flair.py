import os

import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
import argparse

tags_mapping = {
    "question": 1,
    "per_exp": 2,
    "claim": 3,
    "claim_per_exp": 4,
    "O": 0,
}

inverse_tags = {v: k for k, v in tags_mapping.items()}

from src.features.phase1.question_detector import convert_span_to_prediction_format, convert_from_char_to_word

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/Users/wojciechkusa/final-model.pt"
    )
    parser.add_argument("--output_path", type=str, default="data/processed/flair/")
    parser.add_argument(
        "--test_dataset_file", type=str, default="data/interim/st1_test_tokens.txt"
    )
    parser.add_argument("--output_file", type=str, default="st1_test_flair_all.csv")
    parser.add_argument("--reference_file", type=str, default="data/external/st1_public_data/st1_test.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.reference_file, encoding="utf-8")

    model = SequenceTagger.load(args.model_path)

    with open(args.test_dataset_file, "r") as f:
        texts = f.readlines()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_file = f"{args.output_path}/{args.output_file}"

    out_preds = []
    for text in tqdm(texts):
        if text == "[UNK]":
            out_preds.append(np.zeros(1))
            continue

        sentence = Sentence(text)
        # predict the tags
        model.predict(sentence)

        spans = [{"span": x.text, "label": x.tag, "start": x.start_position, "end": x.end_position} for x in sentence.get_spans('ner')]

        y_pred = convert_span_to_prediction_format(text, spans, label_mapping=tags_mapping)
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
                            "subreddit_id": row[1]["subreddit_id"],
                            "post_id": row[1]["post_id"],
                            "words": texts[index_post].split()[index_word],
                            "labels": inverse_tags[int(p)],
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    out_df.to_csv(
        output_file,
        encoding="utf-8",
        index=False,
    )



    # predictions = []
    # for text in texts:
    #     sentence = Sentence(text)
    #     # predict the tags
    #     model.predict(sentence)
    #     print(sentence.to_tagged_string())
    #
    #
    #     for entity in sentence.get_spans('ner'):
    #
    #         # print entity text, start_position and end_position
    #         print(f'entity.text is: "{entity.text}"')
    #         print(f'entity.start_position is: "{entity.start_position}"')
    #         print(f'entity.end_position is: "{entity.end_position}"')
    #
    #         # also print the value and score of its "ner"-label
    #         print(f'entity "ner"-label value is: "{entity.get_label("ner").value}"')
    #         print(f'entity "ner"-label score is: "{entity.get_label("ner").score}"\n')
    #
    #     predictions.append(sentence.to_tagged_string())
    #
    # with open(output_file, "w") as f:
    #     for pred in predictions:
    #         f.write(pred)
    #         f.write("\n")
