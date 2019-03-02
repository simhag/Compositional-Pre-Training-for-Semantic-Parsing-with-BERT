from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from BERT_encoder import BERT
from transformer import Transformer, DecoderLayer, TransformerEncoder, EncoderLayer
from tokens_embeddings import DecoderEmbeddings, PositionalEncoding
import numpy as np

# Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

import random


def initialize_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return


class TSP(nn.Module):
    """ Transformer Semantic Parser:
        - Transformer Encoder
        - Transformer Decoder
    """

    def __init__(self, input_vocab, target_vocab, d_model=512, d_int=2048,  n_layers=6, dropout_rate=0.1):
        """
        :param input_vocab: Vocab based on BERT tokenizer #TODO check if needs more words
        :param target_vocab: Vocab based on BERT tokenizer, requires embedding. Fields tokenizer, tokenizer.ids_to_tokens = ordered_dict
        pad=0, start=1, end=2
        :param size: Size of the BERT model: base or large
        :param d_model: dimension of transformer embeddings #TODO add linear layer to map BERT output to dim 512?
        :param dropout_rate:dropout, default 0.1
        """
        super(TSP, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab  # peutetre faire nous-meme le vocab pour le decoder, puis look-up embedding dessus, petit helper dans le code de hugging face
        self.model_embeddings_source = nn.Sequential(DecoderEmbeddings(vocab=self.input_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=100))
        self.model_embeddings_target = nn.Sequential(DecoderEmbeddings(vocab=self.target_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=100))  # simple look-up embedding for tokens
        # no need for encoder, BERT includes the token embeddings in its architecture
        self.encoder = TransformerEncoder(
            layer=EncoderLayer(d_model=d_model, d_int=d_int, d_k=d_model // 8, d_v=d_model // 8, h=8, p_drop=dropout_rate), n_layer=n_layers)
        self.decoder = Transformer(layer=DecoderLayer(d_model=d_model, d_int=d_int, d_k=d_model // 8, d_v=d_model // 8, h=8, p_drop=0.1), n_layer=n_layers)
        self.linear_projection = nn.Linear(d_model, len(self.target_vocab.tokenizer.ids_to_tokens), bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.device = self.linear_projection.weight.device

        initialize_weights(self.encoder)
        initialize_weights(self.decoder)
        initialize_weights(self.linear_projection)
        initialize_weights(self.model_embeddings_source)
        initialize_weights(self.model_embeddings_target)

    def forward(self, sources: List[str], targets: List[str]) -> torch.Tensor:
        """
        :param source: source strings of size bsize
        :param target: target strings of sizes bsize
        :return: scores, sum of log prob of outputs
        """
        # Take source sentences bsize strings
        # Convert to tokens
        # Keep in minde the nb of tokens per batch example
        # Pad and convert to input tensor for BERT
        source_tokens = self.input_vocab.to_input_tokens(sources)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.model_embeddings_source(self.input_vocab.to_input_tensor(sources, device=self.device))
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(source_tensor, padding_mask=input_padding_mask)  # size batch, maxlen, d_model #no mask right? output c'est un tuple?
        # use lengths kept in mind to get mask over the encoder output (padding mask)
        # Take target and get tokens
        target_tokens = self.target_vocab.to_input_tokens(targets)
        # Add END at the end to get the target we will compare to for log probs
        target_tokens_y = [tokens + ['[END]'] for tokens in target_tokens]
        # Add START at the beginning to get the target we use along with the decoder to generate log probs
        target_tokens = [['[START]'] + tokens for tokens in target_tokens]

        # To be fed to the decoder
        target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens,
                                                                  device=self.device)  # size bsize, max_len
        # To be used for log_probs
        target_y_padded = self.target_vocab.tokens_to_tensor(target_tokens_y, device=self.device)  # size bsize, max_len

        # Mask for the decoder: for padding AND autoregressive constraints
        target_tokens_mask = BSP.generate_target_mask(target_tokens_padded, pad_idx=0)  # size bsize, maxlen, maxlen
        # Ready for the decoder with source, its mask, target, its mask
        decoder_output = self.decode(input_dec=self.model_embeddings_target(target_tokens_padded), output_enc=encoder_output, multihead1_mask=target_tokens_mask, multihead2_mask=input_padding_mask)

        # Projection of the decoder output in linear layer without bias and logsoftmax
        P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)  # size bsize, max_len, len_vocab pour oim

        # Zero out, probabilities for which we have nothing in the target text
        target_masks_y = (target_y_padded != 0).float()

        # Compute log probability of generating true target words -> dark magic I need to check
        target_gold_words_log_prob = torch.gather(P, index=target_y_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks_y
        scores = target_gold_words_log_prob.sum()
        return scores

    def encode(self, source_tensor, padding_mask):
        # simply apply BERT, may need the forward though
        return self.encoder(input_enc=source_tensor, multihead_mask=padding_mask)

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def generate_sent_masks(self, enc_output, source_lengths):
        """
        source_lengths list of ints, len=bsize
        enc_output of size bsize, len, d_model
        :rtype: enc_masks: long tensor of size bsize, len
        """
        enc_masks = torch.ones(enc_output.size(0), enc_output.size(1), dtype=torch.long)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 0
        return enc_masks.to(self.device).unsqueeze(-2)

    @staticmethod
    def generate_target_mask(target_padded, pad_idx):
        """
        target padded = long tensor of size bsize, max_len_tokens
        :rtype: mask of dimension
        """
        tgt_mask = (target_padded != pad_idx).unsqueeze(-2)  # bsize, 1, len
        tgt_mask = tgt_mask & Variable(BSP.subsequent_mask(target_padded.size(-1)).type_as(tgt_mask.data))
        return tgt_mask  # size b, max_len, max_len

    def decode(self, input_dec, output_enc, multihead1_mask, multihead2_mask):
        """
        :param encoder_output: size (b, len, dim_bert)
        :param enc_masks: size (b, len)
        :param target_padded: size (b, len')
        :return:
        """
        return self.decoder(input_dec=input_dec, output_enc=output_enc, multihead1_mask=multihead1_mask,
                            multihead2_mask=multihead2_mask)


class BSP(nn.Module):
    """ BERT Semantic Parser:
        - BERT Encoder
        - Transformer Decoder
    """

    def __init__(self, input_vocab, target_vocab, size='base', d_model=512, dropout_rate=0.1):
        """
        :param input_vocab: Vocab based on BERT tokenizer #TODO check if needs more words
        :param target_vocab: Vocab based on BERT tokenizer, requires embedding. Fields tokenizer, tokenizer.ids_to_tokens = ordered_dict
        pad=0, start=1, end=2
        :param size: Size of the BERT model: base or large
        :param d_model: dimension of transformer embeddings #TODO add linear layer to map BERT output to dim 512?
        :param dropout_rate:dropout, default 0.1
        """
        super(BSP, self).__init__()
        self.hidden_size = 768 if size == 'base' else 1024
        self.dropout_rate = dropout_rate
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab  # peutetre faire nous-meme le vocab pour le decoder, puis look-up embedding dessus, petit helper dans le code de hugging face

        # def whitespace_tokenize(text):
        #     """Runs basic whitespace cleaning and splitting on a piece of text."""
        #     text = text.strip()
        #     if not text:
        #         return []
        #     tokens = text.split()
        #     return tokens
        self.model_embeddings_target = nn.Sequential(DecoderEmbeddings(vocab=self.target_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=100))  # simple look-up embedding for tokens
        # no need for encoder, BERT includes the token embeddings in its architecture
        self.encoder = BERT(size)
        self.decoder = Transformer(layer=DecoderLayer(), N=6)
        self.linear_projection = nn.Linear(d_model, len(self.target_vocab.ids_to_tokens), bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        initialize_weights(self.decoder)
        initialize_weights(self.linear_projection)
        initialize_weights(self.model_embeddings_target)

    def forward(self, sources: List[str], targets: List[str]) -> torch.Tensor:
        """
        :param source: source strings of size bsize
        :param target: target strings of sizes bsize
        :return: scores, sum of log prob of outputs
        """
        # Take source sentences bsize strings
        # Convert to tokens
        # Keep in minde the nb of tokens per batch example
        # Pad and convert to input tensor for BERT
        source_tokens = self.input_vocab.to_input_tokens(sources)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.input_vocab.to_input_tensor_char(sources, device=self.device)
        # feed to BERT
        encoder_output, _ = self.encode(source_tensor)  # size batch, maxlen, d_model
        # use lengths kept in mind to get mask over the encoder output (padding mask)
        enc_masks = self.generate_sent_masks(encoder_output, source_lengths)  # size batch, max_len

        # Take target and get tokens
        target_tokens = self.target_vocab.to_input_tokens(targets)
        # Add END at the end to get the target we will compare to for log probs
        target_tokens_y = [tokens + '[END]' for tokens in target_tokens]
        # Add START at the beginning to get the target we use along with the decoder to generate log probs
        target_tokens = ['[START]' + tokens for tokens in target_tokens]

        # To be fed to the decoder
        target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens,
                                                                  device=self.device)  # size bsize, max_len
        # To be used for log_probs
        target_y_padded = self.target_vocab.tokens_to_tensor(target_tokens_y, device=self.device)  # size bsize, max_len

        # Mask for the decoder: for padding AND autoregressive constraints
        target_tokens_mask = BSP.generate_target_mask(target_tokens_padded, pad_idx=0)  # size bsize, maxlen, maxlen

        # Ready for the decoder with source, its mask, target, its mask
        decoder_output = self.decode(encoder_output, target_tokens_padded, enc_masks, target_tokens_mask)

        # Projection of the decoder output in linear layer without bias and logsoftmax
        P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)  # size bsize, max_len, len_vocab pour oim

        # Zero out, probabilities for which we have nothing in the target text
        target_masks_y = (target_y_padded != 0).float()

        # Compute log probability of generating true target words -> dark magic I need to check
        target_gold_words_log_prob = torch.gather(P, index=target_y_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks_y
        scores = target_gold_words_log_prob.sum()
        return scores

    def encode(self, source_tensor):
        # simply apply BERT, may need the forward though
        return self.encoder(source_tensor)

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def generate_sent_masks(self, enc_output, source_lengths):
        """
        source_lengths list of ints, len=bsize
        enc_output of size bsize, len, d_model
        :rtype: enc_masks: long tensor of size bsize, len
        """
        enc_masks = torch.zeros(enc_output.size(0), enc_output.size(1), dtype=torch.long)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @staticmethod
    def generate_target_mask(target_padded, pad_idx):
        """
        target padded = long tensor of size bsize, max_len_tokens
        :rtype: mask of dimension
        """
        tgt_mask = (target_padded != pad_idx).unsqueeze(-2)  # bsize, 1, len
        tgt_mask = tgt_mask & Variable(BSP.subsequent_mask(target_padded.size(-1)).type_as(tgt_mask.data))
        return tgt_mask  # size b, max_len, max_len

    def decode(self, encoder_output, enc_masks, target_padded, target_padded_mask):
        """
        :param encoder_output: size (b, len, dim_bert)
        :param enc_masks: size (b, len)
        :param target_padded: size (b, len')
        :return:
        """
        return self.decoder(target_padded, encoder_output, multihead1_mask=enc_masks,
                            multihead2_mask=target_padded_mask)

    # def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
    #     Hypothesis]:
    #     """ Given a single source sentence, perform beam search, yielding translations in the target language.
    #     @param src_sent (List[str]): a single source sentence (words)
    #     @param beam_size (int): beam size
    #     @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
    #     @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
    #             value: List[str]: the decoded target sentence, represented as a list of words
    #             score: float: the log-likelihood of the target sentence
    #     """
    #     ## A4 code
    #     # src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
    #     ## End A4 code
    #
    #     src_sents_var = self.vocab.src.to_input_tensor_char([src_sent], self.device)
    #
    #     src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
    #     src_encodings_att_linear = self.att_projection(src_encodings)
    #
    #     h_tm1 = dec_init_vec
    #     att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)
    #
    #     eos_id = self.vocab.tgt['</s>']
    #
    #     hypotheses = [['<s>']]
    #     hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
    #     completed_hypotheses = []
    #
    #     t = 0
    #     while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
    #         t += 1
    #         hyp_num = len(hypotheses)
    #
    #         exp_src_encodings = src_encodings.expand(hyp_num,
    #                                                  src_encodings.size(1),
    #                                                  src_encodings.size(2))
    #
    #         exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
    #                                                                        src_encodings_att_linear.size(1),
    #                                                                        src_encodings_att_linear.size(2))
    #
    #         ## A4 code
    #         # y_tm1 = self.vocab.tgt.to_input_tensor(list([hyp[-1]] for hyp in hypotheses), device=self.device)
    #         # y_t_embed = self.model_embeddings_target(y_tm1)
    #         ## End A4 code
    #
    #         y_tm1 = self.vocab.tgt.to_input_tensor_char(list([hyp[-1]] for hyp in hypotheses), device=self.device)
    #         y_t_embed = self.model_embeddings_target(y_tm1)
    #         y_t_embed = torch.squeeze(y_t_embed, dim=0)
    #
    #         x = torch.cat([y_t_embed, att_tm1], dim=-1)
    #
    #         (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
    #                                             exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
    #
    #         # log probabilities over target words
    #         log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
    #
    #         live_hyp_num = beam_size - len(completed_hypotheses)
    #         contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
    #         top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
    #
    #         prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
    #         hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)
    #
    #         new_hypotheses = []
    #         live_hyp_ids = []
    #         new_hyp_scores = []
    #
    #         decoderStatesForUNKsHere = []
    #         for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
    #             prev_hyp_id = prev_hyp_id.item()
    #             hyp_word_id = hyp_word_id.item()
    #             cand_new_hyp_score = cand_new_hyp_score.item()
    #
    #             hyp_word = self.vocab.tgt.id2word[hyp_word_id]
    #
    #             # Record output layer in case UNK was generated
    #             if hyp_word == "<unk>":
    #                 hyp_word = "<unk>" + str(len(decoderStatesForUNKsHere))
    #                 decoderStatesForUNKsHere.append(att_t[prev_hyp_id])
    #
    #             new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
    #             if hyp_word == '</s>':
    #                 completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
    #                                                        score=cand_new_hyp_score))
    #             else:
    #                 new_hypotheses.append(new_hyp_sent)
    #                 live_hyp_ids.append(prev_hyp_id)
    #                 new_hyp_scores.append(cand_new_hyp_score)
    #
    #         if len(decoderStatesForUNKsHere) > 0 and self.charDecoder is not None:  # decode UNKs
    #             decoderStatesForUNKsHere = torch.stack(decoderStatesForUNKsHere, dim=0)
    #             decodedWords = self.charDecoder.decode_greedy(
    #                 (decoderStatesForUNKsHere.unsqueeze(0), decoderStatesForUNKsHere.unsqueeze(0)), max_length=21,
    #                 device=self.device)
    #             assert len(decodedWords) == decoderStatesForUNKsHere.size()[0], "Incorrect number of decoded words"
    #             for hyp in new_hypotheses:
    #                 if hyp[-1].startswith("<unk>"):
    #                     hyp[-1] = decodedWords[int(hyp[-1][5:])]  # [:-1]
    #
    #         if len(completed_hypotheses) == beam_size:
    #             break
    #
    #         live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
    #         h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
    #         att_tm1 = att_t[live_hyp_ids]
    #
    #         hypotheses = new_hypotheses
    #         hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
    #
    #     if len(completed_hypotheses) == 0:
    #         completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
    #                                                score=hyp_scores[0].item()))
    #
    #     completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    #     return completed_hypotheses
    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.att_projection.weight.device

    # @staticmethod
    # def load(model_path: str, no_char_decoder=False):
    #     """ Load the model from a file.
    #     @param model_path (str): path to model
    #     """
    #     params = torch.load(model_path, map_location=lambda storage, loc: storage)
    #     args = params['args']
    #     model = NMT(vocab=params['vocab'], no_char_decoder=no_char_decoder, **args)
    #     model.load_state_dict(params['state_dict'])
    #
    #     return model
    #
    # def save(self, path: str):
    #     """ Save the odel to a file.
    #     @param path (str): path to the model
    #     """
    #     print('save model parameters to [%s]' % path, file=sys.stderr)
    #
    #     params = {
    #         'args': dict(embed_size=self.model_embeddings_source.embed_size, hidden_size=self.hidden_size,
    #                      dropout_rate=self.dropout_rate),
    #         'vocab': self.vocab,
    #         'state_dict': self.state_dict()
    #     }
    #
    #     torch.save(params, path)
