# Transformers 
from models.phase2.transformer_crf import TRANSFORMERCRF
from transformers import (AutoModel, AutoModelWithLMHead,
                          AutoTokenizer, AutoConfig, AutoModelForTokenClassification)

import sys
path = '/home/anjani/pico-semeval2023/src/models/phase2'
sys.path.append(path)
from models.phase2 import transformer_crf
from transformer_crf import TRANSFORMERCRF

##################################################################################
# Load the chosen tokenizer
##################################################################################
def choose_tokenizer_type(pretrained_model):
    
    if pretrained_model == 'bert':
        tokenizer_ = AutoTokenizer.from_pretrained('bert-base-uncased')
        model_ = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)

    elif pretrained_model == 'roberta':
        tokenizer_ = AutoTokenizer.from_pretrained('roberta-base')
        model_ = AutoModel.from_pretrained('roberta-base', output_hidden_states=True, output_attentions=False)

    elif 'gpt2' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("gpt2", do_lower_case=True, unk_token="<|endoftext|>")
        model_ = AutoModel.from_pretrained("gpt2")

    elif 'biobert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model_ = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    elif 'scibert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        model_ = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

    elif 'roberta' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("roberta-base")
        model_ = AutoModel.from_pretrained("roberta-base")

    elif 'pubmedbert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        model_ = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", output_hidden_states=True, output_attentions=False)

    elif 'BioLinkBERT' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
        model_ = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")

    return tokenizer_ , model_


def choose_model(vector_type, tokenizer, modelembed, chosen_model, args):

    if chosen_model == 'transformercrf':
        model = TRANSFORMERCRF(args.freeze_bert, tokenizer, modelembed, args)