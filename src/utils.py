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


def scaledattention(Q, K, V, mask=None, dropout=None):
    '''
    Compute Scaled attention
    '''
    d_k = Q.size(-1)
    dot_prod = torch.matmul(q, k.transpose(-2, -1))
    dot_prod_scaled = dot_prod / d_k ** 0.5

    if mask is not None:
        dot_prod_scaled.masked_fill_(mask.byte(), -float('inf'))

    attention = F.softmax(dot_prod_scaled, dim=-1)
    if dropout is not None:
        attention = dropout(attention)
    final_output = torch.matmul(attention, V)

    return final_output


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
        for _ in range(max_len - n):
            ids_sequence.append(0) # 0 is the padding id for tokens
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

# if __name__ == '__main__':
#     # file_paths = ['geo880_dev100.tsv', 'geo880_test280.tsv', 'geo880_train100.tsv']
#     # data = read_GeoQuery(['./geoQueryData/' + fpath for fpath in file_paths])
#     # for i, (input_sent, target_query) in enumerate(data_iterator(data, batch_size=2, shuffle=True)):
#     #     if i < 10:
#     #         print(input_sent)
#     #         print(target_query)
#     #         print('---BatchEnd---')
