'''
Define the FeedForward & MultiHead Attention SubLayers used in Encoder and Decoder.
Inspired by the Transformer structure introduced in "Attention Is All You Need" (12/06/2017) - Ashish Vaswani et al. (Google Brain / Google Research)
Date: 02/23
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scaledattention, clones

class FeedForward(nn.Module):
    
    def __init__(self, dim_in_out, dim_int, p_drop):
        '''
        In the paper: dim_in = dim_out = 512, d_int = 2048, p_drop = 0.1
        dim_in_out enforces dim_in to be equal to dim_out, so that the dimensions of the residual connection match
        '''
        super(FeedForward, self).__init__()
        self.dim_in_out = dim_in_out
        self.dim_int = dim_int
        self.p_drop = p_drop

        self.linearlayer1 = nn.Linear(in_features = dim_in_out, out_features = dim_int, bias = True)
        self.linearlayer2 = nn.Linear(in_features = dim_int, out_features = dim_in_out, bias = True)
        self.layernorm = nn.LayerNorm(normalized_shape = dim_in_out)
        self.drop = nn.Dropout(p = p_drop)

    def forward(self, input):
        '''
        Apply FFN(input) = ReLU(input*W1 + b1)*W2 + b2
        Dimension input : (batch_size, dim_in)
        As in the Transformer paper, we add a residual connection around the sublayer followed by layer normalization: output(x) = LayerNorm(x + Sublayer(x))
        '''
        int_relu = F.relu(self.linearlayer1(input))
        #print(int_relu)
        output = self.linearlayer2(int_relu)
        #print(output)
        output_post_drop = self.drop(output)
        #print(output_post_drop)
        final_output = self.layernorm(input + output_post_drop)
        #print(final_output)
        
        return final_output

class MultiHeadAttention(nn.Module):
    
    def __init__(self, h, d_k, d_model, p_drop):
        '''
        In the paper: h = 8, d_k = d_v = 64, d_model = 512, p_drop = 0.1. Assume d_k = d_v
        Check whether d_model is a multiple of h
        '''
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_k
        self.d_model = d_model
        self.p_drop = p_drop

        self.linear_layers = clones(nn.Linear(in_features = d_model, out_features = d_model, bias = False),4)

        self.attention = None
        self.layernorm = nn.LayerNorm(normalized_shape = d_model)
        self.drop = nn.Dropout(p = p_drop)

    def forward(self, Q, K, V, mask = None, verbose = False):
        '''
        Apply a Mutli-Head Attention module
        Dimension Query : (batch_size, len_q, d_model), Keys : (batch_size, len_k, d_model), Values : (batch_size, len_v, d_model)
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W
            where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        As in the Transformer paper, we add a residual connection around the sublayer followed by layer normalization: output(x) = LayerNorm(x + Sublayer(x))
        '''
        if mask is not None:
        # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batches = Q.size(0)

        if verbose:
            print("This is Q input: \n", Q)
            print("Size: ", Q.size())
            print("This is K input: \n", K)
            print("Size: ", K.size())
            print("This is V input: \n", V)
            print("Size: ", V.size())
      
        query, key, value = [layer(k).view(n_batches,-1,self.h,self.d_k).transpose(1,2) for layer,k in zip(self.linear_layers, (Q,K,V))]

        if verbose:
            print("This is Q input: \n", Q)
            print("Size: ", Q.size())
            print("This is K input: \n", K)
            print("Size: ", K.size())
            print("This is V input: \n", V)
            print("Size: ", V.size())

        x, attention = scaledattention(query, key, value, mask = mask, dropout = self.drop, verbose = verbose)
        if verbose:
            print("This is Final Attention: \n", x)
            print("Size: ", x.size())
        self.attention = attention
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        if verbose:
            print("This is Final Attention, after concat: \n", x)
            print("Size: ", x.size())
        output = self.linear_layers[-1](x)
        output_post_drop = self.drop(output)
        final_output = self.layernorm(output_post_drop + Q)

        return final_output
'''
if __name__ == '__main__':
    print('-'*80)
    print('Test FeedForward')
    print('-'*80)

    print('Test on dimensionality')
    dim_input, dim_int, batch_size, p_drop = 20,10,5,0.1
    feedlayer = FeedForward(dim_in_out = dim_input, dim_int = dim_int, p_drop = p_drop)
    x = torch.rand(batch_size, dim_input)
    shape_produced = feedlayer.forward(x).size()
    shape_expected = torch.Size([batch_size, dim_input])
    assert shape_produced == shape_expected, \
        "dimensionality test resulted in shapes {:}, expected {:}".format(shape_produced, shape_expected)
    print('Passed!')

    print('Test on output values')
    dim_input, dim_int, batch_size, p_drop = 4,3,2,0.0
    np.random.seed(123)
    feedlayer = FeedForward(dim_in_out = dim_input, dim_int = dim_int, p_drop = p_drop)
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype = torch.float)
    print("This is X: \n", x)
    print(x.size())
    feedlayer.linearlayer1.weight = nn.Parameter(torch.Tensor([[-1.,1.,0.,2.],[ 0.,1.,1.,1.],[-1.,2.,-1.,-1.]]))
    feedlayer.linearlayer1.bias = nn.Parameter(torch.Tensor([[-1.,-1.,0.],[ 0.,1.,10.]]))
    feedlayer.linearlayer2.weight = nn.Parameter(torch.Tensor([[-1.,2.,0.],[ 0.,1.,1.],[-2.,1.,0.],[5.,2.,3.]]))
    feedlayer.linearlayer2.bias = nn.Parameter(torch.Tensor([[-1.,-1.,0.,5.],[ 0.,1.,10.,-2.]]))
    output_produced = feedlayer.forward(x)
    output_expected = torch.tensor([[-0.4169, -0.3798, -0.8986,  1.6954],
        [-0.3867, -0.4058, -0.9022,  1.6947]])
    print("Output produced: \n", output_produced)
    print("Output expected: \n", output_expected)
    print('Passed!')

    print('-'*80)
    print('Test Multi-Head Attention')
    print('-'*80)

    print('Test on dimensionality')
    d_k,d_model,h,batch_size = 3,6,2,8
    len_q, len_k, len_v = 2,2,2
    multihead = MultiHeadAttention(h = h, d_k = d_k, d_model = d_model, p_drop = 0)
    Q = torch.rand(batch_size, len_q, h*d_k)
    K = torch.rand(batch_size, len_k, h*d_k)
    V = torch.rand(batch_size, len_v, h*d_k)
    shape_produced = multihead.forward(Q,K,V, verbose = True).size()
    shape_expected = torch.Size([batch_size, len_q, d_model])
    assert shape_produced == shape_expected, \
        "dimensionality test resulted in shapes {:}, expected {:}".format(shape_produced, shape_expected)    
    print('Passed!')
'''