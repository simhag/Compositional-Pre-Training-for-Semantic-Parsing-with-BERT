'''
Define the whole Transformer-based Decoder structure
Inspired by "Attention Is All You Need" (12/06/2017) - Ashish Vaswani et al. (Google Brain / Google Research)
Date: 02/26
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sublayers import FeedForward, MultiHeadAttention

class DecoderLayer(nn.Module):
    '''
    Define the DecoderLayer, composed of:
        - MultiHead1 Attention linked to the source sentence
        - MultiHead2 Attention linked to the output of Encoder + MultiHead1
        - FFN linked to MultiHead2
    '''

    def __init__(self, d_model, d_int, d_k, d_v, h, p_drop):
        
        super(DecoderLayer, self).__init__()
        
        self.MultiHead1 = MultiHeadAttention(h = h, d_k = d_k, d_model = d_model, p_drop = p_drop)
        self.MultiHead2 = MultiHeadAttention(h = h, d_k = d_k, d_model = d_model, p_drop = p_drop)
        self.ffn = FeedForward(dim_in_out = d_model, dim_int = d_int, p_drop = p_drop)

    def forward(self, input_dec, output_enc, multihead1_mask, multihead2_mask):
 
        # Assuming Q = K = V = input_dec
        out_head1 = self.MultiHead1.forward(Q = input_dec, K = input_dec, V = input_dec, mask = multihead1_mask)
        out_head2 = self.MultiHead1.forward(Q = out_head1, K = output_enc, V = output_enc, mask = multihead2_mask)

        final_output = self.ffn.forward(out_head2)
        # MASK IN UTILS --> black right i
        # pad_mask + multihead_mask

        return final_output

class Decoder(nn.Module):
    '''
    Generate Decoder, composed of Nx DecoderLayer
        - MultiHead1 Attention linked to the source sentence
        - MultiHead2 Attention linked to the output of Encoder + MultiHead1
        - FFN linked to MultiHead2
    '''

    def __init__(self, n_layer, N):

        super(Decoder, self).__init__()

