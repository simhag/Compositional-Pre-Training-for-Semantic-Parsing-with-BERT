import torch
import os
from utils import data_iterator, get_dataset_finish_by, save_model, get_dataset, load_model, detokenize
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
import domains
from semantic_parser import TSP, BSP
from tensorboardX import SummaryWriter
import time
import math
import numpy as np
from tqdm import tqdm
import data_recombination


def main():
    vocab = Vocab('bert-base-uncased')
    test_inputs = get_dataset_finish_by('geoQueryData', 'train', '_recomb.tsv')
    with open('tokenization_tests.txt', 'w') as test_file:
        test_file.truncate()
        num_matches = 0
        total_examples = 0
        for batch_idx, batch_examples in enumerate(
                data_iterator(test_inputs, batch_size=1, shuffle=False)):
            tokens_list = vocab.to_input_tokens(batch_examples[1])[0]
            detokenized = detokenize(tokens_list)
            if detokenized == batch_examples[1][0]:
                num_matches += 1
            else:
                test_file.write('wrong example:\n')
                test_file.write(batch_examples[1][0] + '\n')
                test_file.write(detokenized + '\n')
                test_file.write('\n' + '-' * 15 + '\n')
            total_examples += 1
        print(f"we obtained the following result: {num_matches / total_examples:.2f} accuracy for detokenization method on given dataset")
    return


def detokenize(tokens_list):
    position = 0
    n = len(tokens_list)
    to_upper_tokens = {'nv'} | {'v' + str(i) for i in range(10)}
    for token in tokens_list:
        if token.startswith('##'):
            tokens_list[position - 1] += token[2:]
            tokens_list = tokens_list[:position] + tokens_list[position + 1:]
            if tokens_list[position - 1] in to_upper_tokens:
                tokens_list[position - 1] = tokens_list[position - 1].upper()
        elif token.startswith('_') and tokens_list[position + 1] != ')':
            tokens_list[position] += tokens_list[position + 1]
            tokens_list = tokens_list[:position + 1] + tokens_list[position + 2:]
        elif token.startswith('\\'):
            tokens_list[position] += tokens_list[position + 1]
            tokens_list = tokens_list[:position + 1] + tokens_list[position + 2:]
        else:
            position += 1
    position = 0
    for i, token in enumerate(tokens_list):
        if token.startswith('_') and tokens_list[position+1].startswith('_'):
            tokens_list[position] += tokens_list[position + 1]
            tokens_list =  tokens_list[:position+1] + tokens_list[position +2:]
        else:
            position += 1
    return ' '.join(tokens_list).replace(' . ', '.')


if __name__ == '__main__':
    main()
