import torch
from argparse import ArgumentParser
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab

parser = ArgumentParser()
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--data_file", type=str, default="geo880_train100.tsv")
parser.add_argument("--BERT", default="bert-base-uncased", type=str, required=True,
                    help="bert-base-uncased, bert-large-uncased")
parser.add_argument("--batch_size", default=32, type=int)


def main(arg_parser):
    test_file_path = os.path.join(arg_parser.data_folder, arg_parser.data_file)
    vocab = Vocab(arg_parser.BERT)
    BERT_encoder = BertModel.from_pretrained(arg_parser.BERT, cache_dir=os.path.join('BERT_pretrained_models', arg_parser.BERT))
    dataset = read_GeoQuery(file_paths=[test_file_path])
    with torch.no_grad():
        for i, (input_sentences_batch, target_queries_batch) in enumerate(data_iterator(dataset, batch_size=arg_parser.batch_size, shuffle=True)):
            if i <100:
                input_tensor = vocab.to_input_tensor(input_sentences_batch, device='cpu')
                output_layers, _ = BERT_encoder(input_tensor, output_all_encoded_layers=False)
                print(output_layers.size())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
