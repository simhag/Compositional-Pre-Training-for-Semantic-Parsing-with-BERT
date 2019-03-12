from collections import namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from BERT_encoder import BERT
from tokens_vocab import Vocab  # for debugging
from tokens_embeddings import DecoderEmbeddings, PositionalEncoding
from torch.autograd import Variable
from transformer import Transformer, DecoderLayer, TransformerEncoder, EncoderLayer

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


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

    def __init__(self, input_vocab, target_vocab, d_model=512, d_int=2048, d_k=64, h=8, n_layers=6, dropout_rate=0.1,
                 max_len_pe=200, bert_name=None):
        """
        :param input_vocab: Vocab based on BERT tokenizer
        :param target_vocab: Vocab based on BERT tokenizer, requires embedding. Fields tokenizer, tokenizer.ids_to_tokens = ordered_dict
        pad=0, start=1, end=2
        :param size: Size of the BERT model: base or large
        :param d_model: dimension of transformer embeddings #TODO add linear layer to map BERT output to dim 512?
        :param dropout_rate:dropout, default 0.1
        """
        super(TSP, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.model_embeddings_source = nn.Sequential(DecoderEmbeddings(vocab=self.input_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=max_len_pe))
        self.model_embeddings_target = nn.Sequential(DecoderEmbeddings(vocab=self.target_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=max_len_pe))
        self.encoder = TransformerEncoder(
            layer=EncoderLayer(d_model=d_model, d_int=d_int, d_k=d_k, d_v=d_k, h=h,
                               p_drop=dropout_rate), n_layer=n_layers)
        self.decoder = Transformer(
            layer=DecoderLayer(d_model=d_model, d_int=d_int, d_k=d_k, d_v=d_k, h=h, p_drop=dropout_rate),
            n_layer=n_layers)
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
        encoder_output = self.encode(source_tensor,
                                     padding_mask=input_padding_mask)  # size batch, maxlen, d_model #no mask right? output c'est un tuple?
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
        target_tokens_mask = TSP.generate_target_mask(target_tokens_padded, pad_idx=0)  # size bsize, maxlen, maxlen
        # Ready for the decoder with source, its mask, target, its mask
        decoder_output = self.decode(input_dec=self.model_embeddings_target(target_tokens_padded),
                                     output_enc=encoder_output, multihead1_mask=target_tokens_mask,
                                     multihead2_mask=input_padding_mask)

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
        tgt_mask = tgt_mask & Variable(TSP.subsequent_mask(target_padded.size(-1)).type_as(tgt_mask.data))
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

    def decode_greedy(self, src_sent, max_len, *args, **kwargs):
        """
        :param src_sent: [ str ] str is the input test example to encode-decode
        :param max_len: max len -in tokens of the input
        :param args:
        :param kwargs:
        :return:list[str] list of the list of tokens for the decoded query
        """
        source_tokens = self.input_vocab.to_input_tokens(src_sent)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.model_embeddings_source(self.input_vocab.to_input_tensor(src_sent, device=self.device))
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(source_tensor,
                                     padding_mask=input_padding_mask)  # size batch, maxlen, d_model #no mask right? output c'est un tuple?
        # use lengths kept in mind to get mask over the encoder output (padding mask)

        target_tokens = [['[START]'] for _ in range(source_tensor.size(0))]
        target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens,
                                                                  device=self.device)  # size bsize, max_len
        target_tokens_mask = TSP.generate_target_mask(target_tokens_padded, pad_idx=0)  # size bsize, maxlen, maxlen
        # Ready for the decoder with source, its mask, target, its mask

        for i in range(max_len - 1):
            decoder_output = self.decode(input_dec=self.model_embeddings_target(target_tokens_padded),
                                         output_enc=encoder_output, \
                                         multihead1_mask=target_tokens_mask, multihead2_mask=input_padding_mask)

            P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)
            _, next_word = torch.max(P[:, -1], dim=-1)

            new_token = self.target_vocab.tokenizer.ids_to_tokens[next_word.item()]
            if new_token == '[END]':
                break
            target_tokens = [tokens + [new_token] for tokens in target_tokens]
            target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens, device=self.device)
            target_tokens_mask = TSP.generate_target_mask(target_tokens_padded, pad_idx=0)
        return [target_token[1:] for target_token in target_tokens]

    def beam_search(self, src_sent, beam_size, max_len):
        len_vocab = len(self.input_vocab.tokenizer.ids_to_tokens)

        source_tokens = self.input_vocab.to_input_tokens(src_sent)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.model_embeddings_source(self.input_vocab.to_input_tensor(src_sent, device=self.device))
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(source_tensor, padding_mask=input_padding_mask)  # size 1, maxlen, d_model

        hypotheses = [['[START]']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        len_hyps = [1]
        hypotheses_padded = self.target_vocab.tokens_to_tensor(hypotheses, device=self.device)
        hyp_tokens_mask = TSP.generate_target_mask(hypotheses_padded, pad_idx=0)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < 2 * beam_size and t < max_len:
            t += 1
            hyp_num = len(hypotheses)

            exp_encoder_output = encoder_output.expand(hyp_num, encoder_output.size(1), encoder_output.size(2))

            decoder_output = self.decode(input_dec=self.model_embeddings_target(hypotheses_padded),
                                         output_enc=exp_encoder_output, multihead1_mask=hyp_tokens_mask,
                                         multihead2_mask=input_padding_mask)

            P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)  # size hyp_num, max_len, len_vocab
            updates = [P[i, len_i - 1, :] for i, len_i in enumerate(len_hyps)]  # n_hyp tensors of size len_vocab
            score_updates = torch.stack(updates)
            continuating_scores = (hyp_scores.unsqueeze(1).expand_as(score_updates) + score_updates).view(
                -1)  # size n_hyp x vocab
            top_scores, top_positions = torch.topk(continuating_scores, k=beam_size)
            prev_hyp_ids = top_positions // len_vocab
            hyp_word_ids = top_positions % len_vocab

            new_hypotheses = []
            new_len_hyps = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_token = self.input_vocab.tokenizer.ids_to_tokens[int(hyp_word_id)]

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_token]
                if hyp_token == '[END]':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],  # on jerte le start et le end
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    new_len_hyps.append(len_hyps[prev_hyp_id] + 1)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) >= 2 * beam_size:
                break

            if len(new_hypotheses) == 0:
                hypotheses = [['[START]']]
                hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
                len_hyps = [1]
            else:
                hypotheses = new_hypotheses
                len_hyps = new_len_hyps
                hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            hypotheses_padded = self.target_vocab.tokens_to_tensor(hypotheses, device=self.device)
            hyp_tokens_mask = TSP.generate_target_mask(hypotheses_padded, pad_idx=0)

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return [hyp.value for hyp in completed_hypotheses[:beam_size]]


class BSP(nn.Module):
    """ BERT Semantic Parser:
        - BERT Encoder
        - Transformer Decoder
    """

    def __init__(self, input_vocab, target_vocab, d_model=512, d_int=2048, d_k=64, h=8, n_layers=6,
                 dropout_rate=0.1, max_len_pe=200, bert_name=None):
        """
        :param input_vocab: Vocab based on BERT tokenizer
        :param target_vocab: Vocab based on BERT tokenizer, requires embedding. Fields tokenizer, tokenizer.ids_to_tokens = ordered_dict
        pad=0, start=1, end=2
        :param size: Size of the BERT model: base or large
        :param d_model: dimension of transformer embeddings
        :param dropout_rate:dropout, default 0.1
        """
        super(BSP, self).__init__()

        self.dropout_rate = dropout_rate
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.hidden_size = 768 if bert_name == 'base' else 1024
        self.encoder = BERT(bert_name=bert_name, d_model=d_model)
        self.model_embeddings_target = nn.Sequential(DecoderEmbeddings(vocab=self.target_vocab, embed_size=d_model),
                                                     PositionalEncoding(d_model=d_model, dropout=dropout_rate,
                                                                        max_len=max_len_pe))
        self.decoder = Transformer(
            layer=DecoderLayer(d_model=d_model, d_int=d_int, d_k=d_k, d_v=d_k, h=h, p_drop=dropout_rate),
            n_layer=n_layers)
        self.linear_projection = nn.Linear(d_model, len(self.target_vocab.tokenizer.ids_to_tokens), bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.device = self.linear_projection.weight.device

        initialize_weights(self.decoder)
        initialize_weights(self.linear_projection)
        initialize_weights(self.model_embeddings_target)

    def forward(self, sources: List[str], targets: List[str]) -> torch.Tensor:
        """
        :param source: source strings of size bsize
        :param target: target strings of sizes bsize
        :return: scores, sum of log prob of outputs
        """
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
        source_tensor = self.input_vocab.to_input_tensor(sources, device=self.device)
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(input_ids=source_tensor, attention_mask=input_padding_mask.squeeze(1))  # size batch, maxlen, d_model #no mask right? output c'est un tuple?
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
        decoder_output = self.decode(input_dec=self.model_embeddings_target(target_tokens_padded),
                                     output_enc=encoder_output, multihead1_mask=target_tokens_mask,
                                     multihead2_mask=input_padding_mask)

        # Projection of the decoder output in linear layer without bias and logsoftmax
        P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)  # size bsize, max_len, len_vocab pour oim

        # Zero out, probabilities for which we have nothing in the target text
        target_masks_y = (target_y_padded != 0).float()

        # Compute log probability of generating true target words -> dark magic I need to check
        target_gold_words_log_prob = torch.gather(P, index=target_y_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks_y
        scores = target_gold_words_log_prob.sum()
        return scores

    def encode(self, input_ids, attention_mask):
        # simply apply BERT, may need the forward though
        return self.encoder(input=input_ids, mask=attention_mask)

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
        tgt_mask = tgt_mask & Variable(TSP.subsequent_mask(target_padded.size(-1)).type_as(tgt_mask.data))
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

    def decode_greedy(self, src_sent, max_len, *args, **kwargs):
        """
        :param src_sent: [ str ] str is the input test example to encode-decode
        :param max_len: max len -in tokens of the input
        :param args:
        :param kwargs:
        :return:list[str] list of the list of tokens for the decoded query
        """
        source_tokens = self.input_vocab.to_input_tokens(src_sent)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.model_embeddings_source(self.input_vocab.to_input_tensor(src_sent, device=self.device))
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(source_tensor,
                                     padding_mask=input_padding_mask)  # size batch, maxlen, d_model #no mask right? output c'est un tuple?
        # use lengths kept in mind to get mask over the encoder output (padding mask)

        target_tokens = [['[START]'] for _ in range(source_tensor.size(0))]
        target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens,
                                                                  device=self.device)  # size bsize, max_len
        target_tokens_mask = TSP.generate_target_mask(target_tokens_padded, pad_idx=0)  # size bsize, maxlen, maxlen
        # Ready for the decoder with source, its mask, target, its mask

        for i in range(max_len - 1):
            decoder_output = self.decode(input_dec=self.model_embeddings_target(target_tokens_padded),
                                         output_enc=encoder_output, \
                                         multihead1_mask=target_tokens_mask, multihead2_mask=input_padding_mask)

            P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)
            _, next_word = torch.max(P[:, -1], dim=-1)

            new_token = self.target_vocab.tokenizer.ids_to_tokens[next_word.item()]
            if new_token == '[END]':
                break
            target_tokens = [tokens + [new_token] for tokens in target_tokens]
            target_tokens_padded = self.target_vocab.tokens_to_tensor(target_tokens, device=self.device)
            target_tokens_mask = TSP.generate_target_mask(target_tokens_padded, pad_idx=0)
        return [target_token[1:] for target_token in target_tokens]

    def beam_search(self, src_sent, beam_size, max_len):
        len_vocab = len(self.input_vocab.tokenizer.ids_to_tokens)

        source_tokens = self.input_vocab.to_input_tokens(src_sent)
        source_lengths = [len(s) for s in source_tokens]
        source_tensor = self.model_embeddings_source(self.input_vocab.to_input_tensor(src_sent, device=self.device))
        # feed to Transformer encoder
        input_padding_mask = self.generate_sent_masks(source_tensor, source_lengths)
        encoder_output = self.encode(source_tensor, padding_mask=input_padding_mask)  # size 1, maxlen, d_model

        hypotheses = [['[START]']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        len_hyps = [1]
        hypotheses_padded = self.target_vocab.tokens_to_tensor(hypotheses, device=self.device)
        hyp_tokens_mask = TSP.generate_target_mask(hypotheses_padded, pad_idx=0)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < 2 * beam_size and t < max_len:
            t += 1
            hyp_num = len(hypotheses)

            exp_encoder_output = encoder_output.expand(hyp_num, encoder_output.size(1), encoder_output.size(2))

            decoder_output = self.decode(input_dec=self.model_embeddings_target(hypotheses_padded),
                                         output_enc=exp_encoder_output, multihead1_mask=hyp_tokens_mask,
                                         multihead2_mask=input_padding_mask)

            P = F.log_softmax(self.linear_projection(decoder_output), dim=-1)  # size hyp_num, max_len, len_vocab
            updates = [P[i, len_i - 1, :] for i, len_i in enumerate(len_hyps)]  # n_hyp tensors of size len_vocab
            score_updates = torch.stack(updates)
            continuating_scores = (hyp_scores.unsqueeze(1).expand_as(score_updates) + score_updates).view(
                -1)  # size n_hyp x vocab
            top_scores, top_positions = torch.topk(continuating_scores, k=beam_size)
            prev_hyp_ids = top_positions // len_vocab
            hyp_word_ids = top_positions % len_vocab

            new_hypotheses = []
            new_len_hyps = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_token = self.input_vocab.tokenizer.ids_to_tokens[int(hyp_word_id)]

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_token]
                if hyp_token == '[END]':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],  # on jerte le start et le end
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    new_len_hyps.append(len_hyps[prev_hyp_id] + 1)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) >= 2 * beam_size:
                break

            if len(new_hypotheses) == 0:
                hypotheses = [['[START]']]
                hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
                len_hyps = [1]
            else:
                hypotheses = new_hypotheses
                len_hyps = new_len_hyps
                hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            hypotheses_padded = self.target_vocab.tokens_to_tensor(hypotheses, device=self.device)
            hyp_tokens_mask = TSP.generate_target_mask(hypotheses_padded, pad_idx=0)

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return [hyp.value for hyp in completed_hypotheses[:beam_size]]


if __name__ == '__main__':
    vocab = Vocab('bert-base-uncased')
    tsp = TSP(input_vocab=vocab, target_vocab=vocab)
    src = 'what is the highest point in florida ?'
    print(vocab.tokenizer.tokenize(src))
