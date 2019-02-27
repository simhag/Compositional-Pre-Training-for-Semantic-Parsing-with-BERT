from pytorch_pretrained_bert.modeling import BertModel
import os


class BERT(object):
    def __init__(self, size='base'):
        if size == 'base':
            path = os.path.join('BERT_pretrained_models', 'bert-base-uncased')
            name = 'bert-base-uncased'
        else:
            path = os.path.join('BERT_pretrained_models', 'bert-large-uncased')
            name = 'bert-large-uncased'

        self.model = BertModel.from_pretrained(name, cache_dir=os.path.join('BERT_pretrained_models', path))

    def forward(self, input_tensor):
        return self.model(input_tensor)
