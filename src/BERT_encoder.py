from pytorch_pretrained_bert.modeling import BertModel
import os
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, size='base', d_model=512):
        super(BERT, self).__init__()
        if size == 'base':
            path = os.path.join('BERT_pretrained_models', 'bert-base-uncased')
            name = 'bert-base-uncased'
        else:
            path = os.path.join('BERT_pretrained_models', 'bert-large-uncased')
            name = 'bert-large-uncased'

        self.model = BertModel.from_pretrained(name, cache_dir=os.path.join('BERT_pretrained_models', path))
        self.output_dim = 768 if size == 'base' else 1024
        self.linear_mapping = nn.Linear(self.output_dim, d_model,
                                        bias=False)  # intuitively, better to have ne bias if only dimension change? Add relu? TODO: ask Robin's advice
        nn.init.xavier_uniform(self.linear_projection.weight)

    def forward(self, input_tensor):
        return self.linear_mapping(self.model(input_tensor))
