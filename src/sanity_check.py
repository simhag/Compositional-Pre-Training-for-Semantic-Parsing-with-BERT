import torch
from argparse import ArgumentParser
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
import domains
from semantic_parser import TSP, BSP
from utils import get_dataset_finish_by, save_model, get_dataset, load_model
from tensorboardX import SummaryWriter
import time
import math
import numpy as np
from tqdm import tqdm
import warnings

parser = ArgumentParser()
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--out_folder", type=str, default="outputs")
parser.add_argument("--BERT", default="bert-base-uncased", type=str, help="bert-base-uncased, bert-large-uncased")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--d_model", default=128, type=int)
parser.add_argument("--d_int", default=512, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--models_path", default='models', type=str)
parser.add_argument("--epoch_to_load", default=40, type=int)
parser.add_argument("--seed", default=1515, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--log_dir", default='logs', type=str)
parser.add_argument("--log", default=True, type=bool)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--save_every", default=5, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--decoding", default='greedy', type=str)
parser.add_argument("--evaluation", default='strict', type=str)
parser.add_argument("--beam_size", default=5, type=int)
parser.add_argument("--max_decode_len", default=105, type=int)
parser.add_argument("--domain", default='geoquery', type=str)

def sanity_check(arg_parser):
    '''
    Check whether the decoding produces [UNK]
    '''
    test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'test','280.tsv')
    vocab = Vocab(arg_parser.BERT)
    file_path = os.path.join(arg_parser.models_path, f"TSP_epoch_{arg_parser.epoch_to_load}.pt")
    model = TSP(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_model,
                n_layers=arg_parser.n_layers, dropout_rate=arg_parser.dropout)
    load_model(file_path=file_path, model=model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    top_parsing_outputs, gold_queries = decoding(model, test_dataset, arg_parser)

    for sentence in top_parsing_outputs:
        if '[UNK]' in sentence[0]:
            warnings.warn('[UNK] in the decoding')

def decoding(loaded_model, test_dataset, arg_parser):
    beam_size = arg_parser.beam_size
    max_len = arg_parser.max_decode_len
    decoding_method = loaded_model.beam_search if arg_parser.decoding == 'beam_search' else loaded_model.decode_greedy
    loaded_model.eval()
    hypotheses = []
    gold_queries = []
    scores = 0
    count = 0
    with torch.no_grad():
        for src_sent_batch, gold_target in tqdm(data_iterator(test_dataset, batch_size=1, shuffle=False), total=280):
            example_hyps = decoding_method(sources=src_sent_batch, max_len=max_len, beam_size=beam_size)
            hypotheses.append(example_hyps)
            gold_queries.append(gold_target[0])
    return hypotheses, gold_queries

if __name__ == '__main__':
    args = parser.parse_args()
    sanity_check(args)
