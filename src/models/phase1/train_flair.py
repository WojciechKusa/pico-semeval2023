from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings

import os

import argparse


def train_ner_model(
    data_folder="data/interim/flair/",
    train_file="st1_train_flair.txt",
    dev_file="st1_val_flair.txt",
    model_name="distilbert-base-uncased",
    model_output_file="data/results/taggers/flair-distilbert",
):
    columns = {0: "text", 1: "ner"}
    label_type = "ner"

    corpus: Corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file=train_file,
        dev_file=dev_file,
    )

    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(train_file, label_dict)

    embeddings = TransformerWordEmbeddings(
        model=model_name,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
        # tag_format="BIO"
    )

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(
        model_output_file,
        learning_rate=5.0e-6,
        max_epochs=25,
        mini_batch_size=32,
        train_with_dev=False,
        mini_batch_chunk_size=1,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_path", type=str, default="data/interim/flair/")
    parser.add_argument(
        "--train_dataset_file", type=str, default="st1_train_flair.txt"
    )
    parser.add_argument("--test_dataset_file", type=str, default="st1_val_flair.txt")
    parser.add_argument(
        "--model_output_file", type=str, default="data/results/taggers/flair-distilbert"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_output_file):
        os.makedirs(args.model_output_file)

    train_ner_model(
        data_folder=args.data_path,
        train_file=args.train_dataset_file,
        dev_file=args.test_dataset_file,
        model_name=args.model_name,
        model_output_file=args.model_output_file,
    )
