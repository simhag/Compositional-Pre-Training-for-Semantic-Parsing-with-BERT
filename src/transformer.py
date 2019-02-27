'''
Define the whole Transformer-based Decoder structure
Inspired by "Attention Is All You Need" (12/06/2017) - Ashish Vaswani et al. (Google Brain / Google Research)
Date: 02/26
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clones
from sublayers import FeedForward, MultiHeadAttention

class DecoderLayer(nn.Module):
    '''
    Define the DecoderLayer, composed of:
        - MultiHead1 Attention linked to the source sentence
        - MultiHead2 Attention linked to the output of Encoder + MultiHead1
        - FFN linked to MultiHead2
    '''

    def __init__(self, d_model = 512, d_int = 2048, d_k = 64, d_v = 64, h = 8, p_drop = 0.1):
        
        super(DecoderLayer, self).__init__()
        
        self.MultiHead1 = MultiHeadAttention(h = h, d_k = d_k, d_model = d_model, p_drop = p_drop)
        self.MultiHead2 = MultiHeadAttention(h = h, d_k = d_k, d_model = d_model, p_drop = p_drop)
        self.ffn = FeedForward(dim_in_out = d_model, dim_int = d_int, p_drop = p_drop)

    def forward(self, input_dec, output_enc, multihead1_mask = None, multihead2_mask = None):
        # Multiheadmask set as None by default: to be changed !!!
 
        out_head1 = self.MultiHead1(Q = input_dec, K = input_dec, V = input_dec, mask = multihead1_mask)
        out_head2 = self.MultiHead2(Q = out_head1, K = output_enc, V = output_enc, mask = multihead2_mask)

        final_output = self.ffn(out_head2)
        # MASK IN UTILS --> black right i
        # pad_mask + multihead_mask

        return final_output

class Transformer(nn.Module):
    '''
    Generate Decoder, composed of Nx DecoderLayer
    '''

    def __init__(self, layer, n_layer = 6):

        super(Transformer, self).__init__()
        self.layers_decoder = clones(module = layer, N = n_layer)


    def forward(self, input_dec, output_enc):

        for layer in self.layers_decoder:
            input_dec = layer(input_dec = input_dec, output_enc = output_enc)
            # ATTENTION: ADD MASK

        return input_dec

'''
if __name__ == '__main__':
    d_model, batch_size, d_int, d_k, d_v, h, len_output_sent = 6, 8, 3, 3, 3, 2, 10
    input_dec = torch.rand([batch_size, len_output_sent, d_model])
    output_enc = torch.rand([batch_size, len_output_sent, d_model])
    decoder_layer = DecoderLayer(d_model = d_model, d_int = d_int, d_k = d_k, d_v = d_v, h = h, p_drop = 0.1)
    transformer_test = Transformer(layer = decoder_layer, n_layer = 6)
    print(transformer_test(input_dec = input_dec, output_enc = output_enc).size())
'''