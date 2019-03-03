import torch
from argparse import ArgumentParser
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
from semantic_parser import TSP, BSP
from utils import get_dataset, save_model, load_model
from tensorboardX import SummaryWriter
import time
import math
import numpy as np

parser = ArgumentParser()
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--out_folder", type=str, default="outputs")
parser.add_argument("--BERT", default="bert-base-uncased", type=str, help="bert-base-uncased, bert-large-uncased")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--d_model", default=128, type=int)
parser.add_argument("--d_int", default=512, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--models_path", default='models', type=str)
parser.add_argument("--epoch_to_load", default=40, type=int)
parser.add_argument("--seed", default=1515, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--log_dir", default='logs', type=str)
parser.add_argument("--log", default=True, type=bool)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--save_every", default=5, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--decoding", default='greedy', type=str)
parser.add_argument("--evaluation", default='strict', type=str)
parser.add_argument("--beam_size", default=5, type=int)
parser.add_argument("--max_decode_len", default=105, type=int)


def main(arg_parser):
    # seed the random number generators
    seed = arg_parser.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    if arg_parser.train:
        train(arg_parser)
    if arg_parser.test:
        test(arg_parser)
    return


def train(arg_parser):
    train_dataset = get_dataset(arg_parser.data_folder, 'train')
    test_dataset = get_dataset(arg_parser.data_folder, 'dev')
    vocab = Vocab(arg_parser.BERT)
    model = TSP(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_model,
                n_layers=arg_parser.n_layers, dropout_rate=arg_parser.dropout)

    model.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=arg_parser.lr)

    if not os.path.isdir(arg_parser.log_dir):
        os.makedirs(arg_parser.log_dir)

    summary_writer = SummaryWriter(log_dir=arg_parser.log_dir) if arg_parser.log else None

    n_train = len(train_dataset)
    n_test = len(test_dataset)

    for epoch in range(arg_parser.epochs):
        running_loss = 0.0
        last_log_time = time.time()

        # Training
        train_loss = 0.0
        for batch_idx, batch_examples in enumerate(
                data_iterator(train_dataset, batch_size=arg_parser.batch_size, shuffle=arg_parser.shuffle)):
            if ((batch_idx % 100) == 0) and batch_idx > 1:
                print("epoch {} | batch {} | mean running loss {:.2f} | {:.2f} batch/s".format(epoch, batch_idx,
                                                                                               running_loss / 100,
                                                                                               100 / (
                                                                                                       time.time() - last_log_time)))
                last_log_time = time.time()
                running_loss = 0.0

            sources, targets = batch_examples[0], batch_examples[1]
            example_losses = -model(sources, targets)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / arg_parser.batch_size

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arg_parser.clip_grad)

            # if batch_idx % 100 == 0:
            #     print("{:.2f}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss
            running_loss += loss.item()
            train_loss += loss.item()

        print("Epoch train loss : {}".format(math.sqrt(train_loss / math.ceil(n_train / arg_parser.batch_size))))

        if summary_writer is not None:
            summary_writer.add_scalar("train/loss",
                                      train_loss / math.ceil(n_train / arg_parser.batch_size),
                                      global_step=epoch)
        if (epoch % arg_parser.save_every == 0) and arg_parser.log and epoch > 0:
            save_model(arg_parser.models_path, "{}_epoch_{}.pt".format('TSP', epoch), model,
                       device)
        ## TEST
        test_loss = 0.0

        for batch_idx, batch_examples in enumerate(
                data_iterator(test_dataset, batch_size=arg_parser.batch_size,
                              shuffle=arg_parser.shuffle)):
            with torch.no_grad():
                sources, targets = batch_examples[0], batch_examples[1]
                example_losses = -model(sources, targets)  # (batch_size,)
                batch_loss = example_losses.sum()
                loss = batch_loss / arg_parser.batch_size

                test_loss += loss.item()

        if summary_writer is not None:
            summary_writer.add_scalar("test/loss", test_loss / math.ceil(n_test / arg_parser.batch_size),
                                      global_step=epoch)
        print("TEST loss | epoch {} | {:.2f}".format(epoch, test_loss / math.ceil(n_test / arg_parser.batch_size)))

    return None


# TODO beam_search and knowledge_based eval
def test(arg_parser):
    test_dataset = get_dataset(arg_parser.data_folder, 'test')
    vocab = Vocab(arg_parser.BERT)
    file_path = os.path.join(arg_parser.models_path, f"TSP_epoch_{arg_parser.epoch_to_load}.pt")
    model = TSP(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_model,
                n_layers=arg_parser.n_layers, dropout_rate=arg_parser.dropout)
    load_model(file_path=file_path, model=model)
    evaluation_methods = {'strict': strict_evaluation, 'soft': knowledge_based_evaluation}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    parsing_outputs, gold_queries = decoding(model, test_dataset, arg_parser)

    top_parsing_outputs = [parsing_output[0] for parsing_output in parsing_outputs]

    test_accuracy = evaluation_methods[arg_parser.evaluation](top_parsing_outputs, gold_queries)
    print(
        f"test accuracy for model {arg_parser.evaluation}_{arg_parser.decoding}_TSP_{arg_parser.epoch_to_load}: {test_accuracy:2f}")

    outfile = os.path.join(arg_parser.data_folder, os.path.join(arg_parser.out_folder,
                                                                f"{arg_parser.evaluation}_{arg_parser.decoding}_TSP_{arg_parser.epoch_to_load}.txt"))
    with open(outfile, 'w') as f:
        for parsing_output in top_parsing_outputs:
            f.write(parsing_output + '\n')
    return None


def decoding(loaded_model, test_dataset, arg_parser):
    beam_size = arg_parser.beam_size
    max_len = arg_parser.max_decode_len
    decoding_method = loaded_model.beam_search if arg_parser.decoding == 'beam_search' else loaded_model.decode_greedy
    loaded_model.eval()
    hypotheses = []
    gold_queries = []
    with torch.no_grad():
        for src_sent_batch, gold_target in data_iterator(test_dataset, batch_size=1, shuffle=False):
            example_hyps = src_sent_batch#decoding_method(src_sent_batch[0], max_len, beam_size)
            hypotheses.append(example_hyps)
            gold_queries.append(gold_target[0])
    return hypotheses, gold_queries


def strict_evaluation(model_queries, gold_queries):
    n = len(model_queries)
    assert n == len(gold_queries)
    return sum([model_queries[x] == gold_queries[x] for x in range(n)]) / n


def knowledge_based_evaluation(model_queries, gold_queries):
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    # train(args)
    test(args)