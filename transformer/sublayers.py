'''
Define the FeedForward & MultiHead Attention SubLayers used in Encoder and Decoder.
Inspired by the Transformer structure introduced in "Attention Is All You Need" (12/06/2017) - Ashish Vaswani et al. (Google Brain / Google Research)
Date: 02/23
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class 

'''
if __name__ == '__main__':
    print('-'*80)
    print('Test FeedForward')
    print('-'*80)

    print('Running test on FeedForward!')
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
    print("This is X: ", x)
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
    print('Multi-Head Attention')
    print('-'*80)
'''
