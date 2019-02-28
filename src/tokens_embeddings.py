import torch.nn as nn
import torch
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    #TODO check the numerical values given our small input lengths...
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input: b, len, embed_size
        pe of size (1, len, embed_size)
        :rtype: b, len, embed_size
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class DecoderEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, vocab, embed_size=512, dropout_rate=0.1, max_len=100):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(DecoderEmbeddings, self).__init__()
        pad_token_idx = vocab.ids_to_tokens['[PAD]']
        assert pad_token_idx == 0
        self.embeddings = nn.Embedding(len(vocab.ids_to_tokens), embed_size, padding_idx=pad_token_idx)
        self.positional_encoding = PositionalEncoding(d_model=embed_size, dropout=dropout_rate, max_len=max_len)

    def forward(self, input):
        """
        input: size bsize, max_n_tokens
        :rtype: size bsize, max_n_tokens, embed_size
        """
        output = self.embeddings(input)
        return output
