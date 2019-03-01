from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import pad_ids_sequences
import torch


# TODO verif que le dataset ne contienne pas de mots inconnus, auquel cas les rajouter
class Vocab(object):
    def __init__(self, BERT_model):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_model, do_lower_case=True)
        # CUSTOM START AND END TOKENS ADDED
        self.tokenizer.ids_to_tokens[1] = '[START]'
        self.tokenizer.ids_to_tokens[2] = '[END]'
        # self.tokenizer.ids_to_tokens[3] = '</s>'

    def to_input_tensor(self, input_strings_sequences, device):
        """
        :param input_strings_sequences: list[string] batch of input question strings, len=bsize
        :param device: torch device
        :return: tensor of size (bsize, max_input_len), tensor of token ids, max_input_len is the max_len (in tokens) of input questions in the batch
        """
        token_sequences = [self.tokenizer.tokenize(sequence) for sequence in input_strings_sequences]
        token_ids_sequences = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in token_sequences]
        padded_ids_sequences = pad_ids_sequences(token_ids_sequences)
        return torch.tensor(padded_ids_sequences, dtype=torch.long, device=device)

    def to_input_tokens(self, input_strings_sequences):
        return [self.tokenizer.tokenize(sequence) for sequence in input_strings_sequences]

    def tokens_to_tensor(self, input_tokens_sequences, device):
        token_ids_sequences = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in input_tokens_sequences]
        padded_ids_sequences = pad_ids_sequences(token_ids_sequences)
        return torch.tensor(padded_ids_sequences, dtype=torch.long, device=device)
