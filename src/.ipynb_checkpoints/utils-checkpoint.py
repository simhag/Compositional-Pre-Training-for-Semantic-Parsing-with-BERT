'''
Utils functions for Transformer structure
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
        dot_prod_scaled.masked_fill_(mask.byte(), -float('inf'))

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

'''
if __name__ == '__main__':
    print('-'*80)
    print('Test clones')
    print('-'*80)
    module = nn.Linear(in_features = 5, out_features = 3, bias = False)
    module_list = clones(module, 4)
    print("Module List : \n", module_list)
'''