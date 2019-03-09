import torch
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
import domains
from semantic_parser import TSP, BSP
from utils import get_dataset_finish_by, save_model, get_dataset, load_model
from tensorboardX import SummaryWriter
import time
import math
import numpy as np
from tqdm import tqdm
import data_recombination


def main():
    vocab = Vocab('bert-base-uncased')
    test_inputs = 

    return

if __name__ == '__main__':
    main()