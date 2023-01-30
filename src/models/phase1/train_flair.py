from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings

import os

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--results_path", type=str, default="results/flair/")

    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK']='True' # fixme: remove this line

    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = 'data/interim/flair/'

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='st1_train_flair.txt',
                                  dev_file='st1_train_flair.txt')

    print(len(corpus.train))
    print(corpus.train[0].to_tagged_string('ner'))


    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)



    model_name = "distilbert-base-uncased"

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model=model_name,
                                           layers="-1",
                                           subtoken_pooling="first",
                                           fine_tune=True,
                                           use_context=True,
                                           )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=True,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )

    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)


    # 7. run fine-tuning
    trainer.fine_tune('data/results/taggers/flair-distilbert',
                      learning_rate=5.0e-4,
                      max_epochs=30,
                      mini_batch_size=32,
                      mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                      )
