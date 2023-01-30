from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

from flair.data import Sentence
from flair.models import SequenceTagger
# load the trained model


import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/flair/final-model.pt")

    args = parser.parse_args()

    model = SequenceTagger.load(args.model_path)
    # create example sentence

    test_dataset_file = "data/interim/st1_test_tokens.txt"
    with open(test_dataset_file, "r") as f:
        texts = f.readlines()


    output_file = "results/flair/st1_test_flair.txt"
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
