import torch
from argparse import ArgumentParser
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
from semantic_parser import TSP, BSP
from utils import get_dataset_finish_by, save_model
from tensorboardX import SummaryWriter
import time
import math

parser = ArgumentParser()
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--data_file", type=str, default="geo880_train100.tsv")
parser.add_argument("--BERT", default="bert-base-uncased", type=str, help="bert-base-uncased, bert-large-uncased")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--d_model", default=128, type=int)
parser.add_argument("--d_int", default=512, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--models_path", default='models', type=str)
parser.add_argument("--seed", default=1515, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--log_dir", default='logs', type=str)
parser.add_argument("--log", default=True, type=bool)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--save_every", default=5, type=int)
parser.add_argument("--n_layers", default=2, type=int)


def main(arg_parser):
    pass


def train(arg_parser):
    train_dataset = get_dataset_finish_by(arg_parser.data_folder, 'train','600.tsv')
    test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'dev', '100.tsv')
    vocab = Vocab(arg_parser.BERT)
    model = TSP(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_model, n_layers=arg_parser.n_layers, dropout_rate=arg_parser.dropout)

    model.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=arg_parser.lr)

    # dataset = read_GeoQuery(file_paths=test_file_paths)
    # with torch.no_grad():
    #     for i, (input_sentences_batch, target_queries_batch) in enumerate(
    #             data_iterator(dataset, batch_size=arg_parser.batch_size, shuffle=True)):
    #         input_tensor = vocab.to_input_tensor(input_sentences_batch, device='cpu')
    #         output_layers, _ = BERT_encoder(input_tensor, output_all_encoded_layers=False)
    #         # output_layers of dim bsize, max_len, 768 or 1024 (base or large)
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
                                                                                               math.sqrt(
                                                                                                   running_loss / 100),
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

            if batch_idx % 100 == 0:
                print("{:.2f}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss
            running_loss += loss.item()
            train_loss += loss.item()

        print("Epoch train loss : {}".format(math.sqrt(train_loss / math.ceil(n_train / arg_parser.batch_size))))

        if summary_writer is not None:
            summary_writer.add_scalar("train/loss",
                                      math.sqrt(train_loss / math.ceil(n_train / arg_parser.batch_size)),
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
            summary_writer.add_scalar("test/loss", math.sqrt(test_loss / math.ceil(n_test) / arg_parser.batch_size),
                                      global_step=epoch)
        print("TEST loss | epoch {} | {:.2f}".format(epoch, math.sqrt(
            test_loss / math.ceil(n_test / arg_parser.batch_size))))

    return None

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
