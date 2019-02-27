'''
Utils functions for Transformer structure
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import csv
import math
import random


def scaledattention(Q, K, V, mask = None, dropout = None, verbose = False):
    '''
    Compute Scaled attention
    '''
    
    d_k = Q.size(-1)
    dot_prod = torch.matmul(Q,K.transpose(-2,-1))
    if verbose:
        print("This is QK^T: \n", dot_prod)
        print("Size: ", dot_prod.size())
    dot_prod_scaled = dot_prod/d_k**0.5
    
    if mask is not None:
        dot_prod_scaled  = dot_prod_scaled.masked_fill_(mask.transpose(2,3) == 0, -float('inf'))

    attention = F.softmax(dot_prod_scaled, dim = -1)
    if verbose:
        print("This is attention: \n", attention)
        print("Size: ", attention.size())

    if dropout is not None:
    	attention = dropout(attention)
    final_output = torch.matmul(attention, V)

    return final_output, attention


def clones(module, N):
    '''
    Produce a list of modules
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def pad_ids_sequences(ids_sequences):
    """
    :param ids_sequences: list[list[int]] batch of token ids list, one for each bactch example
    :return: padded sequences with 0
    """
    max_len = max([len(ids_sequence) for ids_sequence in ids_sequences])
    for ids_sequence in ids_sequences:
        # if 0 in ids_sequence:
        #     print('Excuse me wtf?')
        n = len(ids_sequence)
        ids_sequence += [0] * (max_len - n)
        # assert len(ids_sequence) == max_len
    return ids_sequences


# TODO ask robin about spaces in target outputs
def read_GeoQuery(file_paths):
    """
    :param file_path:list of paths to geoquery files
    :return: data: list of string tuples input_sentence, output_geoquery from the tsv.file
    """
    data = []
    for geopath in file_paths:
        with open(geopath, 'r') as geofile:
            reader = csv.reader(geofile, delimiter="\t")
            for row in reader:
                data.append((row[0], row[1]))
    return data


def data_iterator(input_data, batch_size, shuffle=False):
    """
    Iterator over the input dataset yielding batches of input,output examples
    :param input_data: list of input, output tuples in the geoquery dataset,as returned by read_GeoQuery
    :param batch_size: int
    :param shuffle: shuffle dataset or not
    """
    n_inputs = len(input_data)
    num_batches = math.ceil(n_inputs // batch_size)
    indexes_list = list(range(n_inputs))

    if shuffle:
        random.shuffle(indexes_list)

    for i in range(num_batches):
        batch_indexes = indexes_list[i * batch_size: (i + 1) * batch_size]
        batch_examples = [input_data[index] for index in batch_indexes]
        batch_examples = sorted(batch_examples, key=lambda x: len(x[0]),
                                reverse=True)  # sort input sequences with first sequence of max_len
        inputs = [batch_example[0] for batch_example in batch_examples]
        outputs = [batch_example[1] for batch_example in batch_examples]
        yield (inputs, outputs)

'''
def generate_subsequent_mask(seq):
    
    Mask the subsequent info 
    
    att_mask = 
'''

def generate_sent_mask(tgt, source_lengths):
    '''
    Generate sentence masks for encoder hidden states
    :param: tensor of shape (batch_size, max_source_len)
    :return: tensor of dimensions (batch_size, max_source_len) containing 1 in positions corresponding to 'pad' tokens and 0 for non-pad tokens
    '''
    tgt_mask = torch.zeros(tgt.size(0), tgt.size(1), dtype = torch.float)
    for e_id, src_len in enumerate(source_lengths):
        tgt_mask[e_id, src_len:] = 1
    return tgt_mask.to()


# if __name__ == '__main__':
#     # file_paths = ['geo880_dev100.tsv', 'geo880_test280.tsv', 'geo880_train100.tsv']
#     # data = read_GeoQuery(['./geoQueryData/' + fpath for fpath in file_paths])
#     # for i, (input_sent, target_query) in enumerate(data_iterator(data, batch_size=2, shuffle=True)):
#     #     if i < 10:
#     #         print(input_sent)
#     #         print(target_query)
#     #         print('---BatchEnd---')
