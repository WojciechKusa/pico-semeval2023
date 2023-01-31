"""Merges predictions from different outputs into a single file"""
import copy
from typing import List

import pandas as pd


tags = {  # tag and priority
    "question": 1,
    "per_exp": 2,
    "claim": 3,
    "claim_per_exp": 4,
    "O": 0,
}

def resolve_tie(tags: List[str]) -> str:
    """Resolves ties in predictions by
    1. choosing the tag which is the most common
    2. if there is still tie, then choose the tag with the highest priority"""
    if len(tags) == 1:
        return tags[0]
    else:
        # get the most common tag
        tag = max(set(tags), key=tags.count)
        # if there is still a tie, then choose the tag with the highest priority
        if tags.count(tag) > 1:
            tag = max(tags, key=lambda x: tags.index(x))
        return tag


def merge_predictions(predictions_files: List[str]) -> pd.DataFrame:
    """load and iterate over predictions files and merge them into a single file"""

    predictions = []
    for file in predictions_files:
        df = pd.read_csv(file)
        predictions.append(df)

    # check if all have same length
    assert len({len(df) for df in predictions}) == 1

    out_df = copy.deepcopy(predictions[0])
    for i in range(len(predictions[0])):
        tags = [df.loc[i, "labels_char"] for df in predictions]
        out_df.loc[i, "labels_char"] = resolve_tie(tags)

    return out_df


if __name__ == '__main__':
    prediction_files = [
        "data/processed/question_detector/st1_test_predictions.csv",
    ]

    out_df = merge_predictions(prediction_files)
    out_df.to_csv("data/processed/submission/st1_test_predictions_merged.csv", index=False)