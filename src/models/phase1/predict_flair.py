import os
from flair.data import Sentence
from flair.models import SequenceTagger

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="results/flair/final-model.pt"
    )
    parser.add_argument("--results_path", type=str, default="results/flair/")
    parser.add_argument(
        "--test_dataset_file", type=str, default="data/interim/st1_test_tokens.txt"
    )
    parser.add_argument("--output_file", type=str, default="st1_test_flair.txt")

    args = parser.parse_args()

    model = SequenceTagger.load(args.model_path)

    with open(args.test_dataset_file, "r") as f:
        texts = f.readlines()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_file = f"{args.output_path}/{args.output_file}"
    predictions = []
    for text in texts:
        sentence = Sentence(text)
        # predict the tags
        model.predict(sentence)
        print(sentence.to_tagged_string())

        predictions.append(sentence.to_tagged_string())

    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(pred)
            f.write("\n")
