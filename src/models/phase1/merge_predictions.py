"""Merges predictions from different outputs into a single file"""
import copy
from typing import List
from tqdm import tqdm
import pandas as pd
from collections import Counter


tags_mapping = {  # tag and priority
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
        c = Counter(tags)
        most_common = [x for x in c if c[x] == c.most_common(1)[0][1]]
        if len(most_common) == 1:
            return most_common[0]
        else:
            tag = max(tags, key=lambda x: tags_mapping[x])
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
    for i in tqdm(range(len(predictions[0]))):
        tags = [df.loc[i, "labels"] for df in predictions]
        out_df.loc[i, "labels"] = resolve_tie(tags)

    return out_df


if __name__ == '__main__':
    prediction_files = [
        "data/processed/question_detector/st1_test_predictions.csv",
        "data/processed/flair/flair_submission.csv",
    ]

    out_df = merge_predictions(prediction_files)
    out_df.to_csv("data/processed/submission/st1_test_predictions_merged.csv", index=False)