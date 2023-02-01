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
    0. if there is only one tag, then return it
    1. per_exp + claim == claim_per_exp
    2. choosing the tag which is the most common
    3. if there is still tie, then choose the tag with the highest priority
    """
    if len(tags) == 1:
        return tags[0]
    else:
        # per_exp + claim == claim_per_exp
        if "per_exp" in tags and "claim" in tags:
            return "claim_per_exp"
        # remove 'O' tag
        tags = [x for x in tags if x != "O"]
        if len(tags) == 0:
            return "O"
        # get the most common tag
        c = Counter(tags)
        most_common = [x for x in c if c[x] == c.most_common(1)[0][1]]
        if len(most_common) == 1:
            return most_common[0]
        else:
            # if there is still a tie, then choose the tag with the highest priority
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
    # prediction_files = [
    #     "data/processed/question_detector/st1_test_predictions.csv",
    #     # "data/processed/flair/all_flair_submission.csv",
    #     "data/processed/flair/question_flair_submission.csv",
    # ]
    #
    # prediction_files = [ # run 1
    #     "data/processed/submission/st1_test_predictions_merged.csv",
    #     "data/processed/flair/st1_test_flair_per_exp.csv"
    # ]
    #
    # prediction_files = [  # run 2
    #     "data/processed/flair/all_flair_submission.csv",
    # "data/processed/submission/st1_test_predictions_question_perexp-total.csv"
    # ]
    #
    #
    prediction_files = [ # run 3
    "data/processed/submission/st1_test_predictions_question_perexp-total.csv",
    "data/processed/flair/st1_test_flair_roberta.csv",
    ]


    # prediction_files = [ # run 4
    # "data/processed/flair/st1_test_flair_roberta.csv",
    # "data/processed/submission/run_2.csv"
    # ]



    out_df = merge_predictions(prediction_files)
    out_df.to_csv("data/processed/submission/run_3.csv", index=False)