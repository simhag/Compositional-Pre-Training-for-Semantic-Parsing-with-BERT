import torch
from argparse import ArgumentParser
import os
from utils import data_iterator, get_dataset_finish_by, save_model, load_model, detokenize
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
import domains
from optimizer import NoamOpt
from semantic_parser import TSP, BSP
from tensorboardX import SummaryWriter
import time
import math
import numpy as np
from tqdm import tqdm
import data_recombination

DOMAIN = 'geoquery'

parser = ArgumentParser()
# FOLDERS
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--out_folder", type=str, default="outputs")
parser.add_argument("--log_dir", default='logs', type=str)
parser.add_argument("--subdir", default="run1", type=str)
parser.add_argument("--models_path", default='models', type=str)
# MODEL
parser.add_argument("--TSP_BSP", default=0, type=int, help="1: TSP model, 0:BSP")
parser.add_argument("--BERT", default="base", type=str, help="bert-base-uncased or bert-large-uncased (large)")
# MODEL PARAMETERS
parser.add_argument("--d_model", default=128, type=int)
parser.add_argument("--d_int", default=128, type=int)
parser.add_argument("--h", default=8, type=int)
parser.add_argument("--d_k", default=16, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--max_len_pe", default=200, type=int)
# DATA RECOMBINATION
parser.add_argument("--recombination_method", default='entity', type=str)
parser.add_argument("--extras_train", default=600, type=int)
parser.add_argument("--extras_dev", default=100, type=int)
# TRAINING PARAMETERS
parser.add_argument("--train_arg", default=0, type=int)
parser.add_argument("--train_load", default=0, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--optimizer", default=1, type=int)
parser.add_argument("--warmups_steps", default=4000, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--save_every", default=50, type=int)
parser.add_argument("--log", default=True, type=bool)
parser.add_argument("--shuffle", default=True, type=bool)
# TESTING PARAMETERS
parser.add_argument("--test_arg", default=1, type=int)
parser.add_argument("--epoch_to_load", default=55, type=int)
parser.add_argument("--decoding", default='beam_search', type=str)
parser.add_argument("--beam_size", default=5, type=int)
parser.add_argument("--max_decode_len", default=250, type=int)
# RANDOM SEED
parser.add_argument("--seed", default=1515, type=int)


def do_data_recombination(argparser):
    folders = ['train', 'dev', 'test']
    nums_folder = {'train': argparser.extras_train, 'dev': argparser.extras_dev, 'test': 0}
    global DOMAIN
    for folder in folders:
        data_recombination.main(folder=folder, domain=DOMAIN, augmentation_type=argparser.recombination_method,
                                num=nums_folder[folder])
    return


def main(arg_parser):
    assert arg_parser.h * arg_parser.d_k == arg_parser.d_model, "d_k * h must be equal to d_model"
    # create_directories(arg_parser)
    # if not os.path.isdir(arg_parser.log_dir):
    #     os.makedirs(arg_parser.log_dir)
    seed = arg_parser.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    do_data_recombination(arg_parser)
    if arg_parser.train_arg:
        train(arg_parser)
    if arg_parser.test_arg:
        test(arg_parser)
    return


def get_model_name(argparser):
    if argparser.TSP_BSP:
        base_model = 'TSP'
    else:
        base_model = 'BSP'
    return base_model # f"{base_model}_d_model{argparser.d_model}_layers{argparser.n_layers}_recomb{argparser.recombination_method}_extrastrain{argparser.extras_train}_extrasdev{argparser.extras_dev}"


def train(arg_parser):
    logs_path = os.path.join(arg_parser.log_dir, arg_parser.subdir)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    file_name_epoch_indep = get_model_name(arg_parser)
    recombination = arg_parser.recombination_method

    vocab = Vocab(f'bert-{arg_parser.BERT}-uncased')
    model_type = TSP if arg_parser.TSP_BSP else BSP
    model = model_type(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_int,
                       d_k=arg_parser.d_k, h=arg_parser.h, n_layers=arg_parser.n_layers,
                       dropout_rate=arg_parser.dropout, max_len_pe=arg_parser.max_len_pe, bert_name=arg_parser.BERT)

    file_path = os.path.join(arg_parser.models_path, f"{file_name_epoch_indep}_epoch_{arg_parser.epoch_to_load}.pt")
    if arg_parser.train_load:
        train_dataset = get_dataset_finish_by(arg_parser.data_folder, 'train',
                                              f"{600}_{recombination}_recomb.tsv")
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'dev',
                                             f"{100}_{recombination}_recomb.tsv")
        load_model(file_path=file_path, model=model)
        print('loaded model')
    else:
        train_dataset = get_dataset_finish_by(arg_parser.data_folder, 'train',
                                              f"{600 + arg_parser.extras_train}_{recombination}_recomb.tsv")
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'dev',
                                             f"{100 + arg_parser.extras_dev}_{recombination}_recomb.tsv")
    model.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if arg_parser.optimizer:
        optimizer = NoamOpt(model_size=arg_parser.d_model, factor=1, warmup=arg_parser.warmups_steps, \
                            optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=arg_parser.lr)
    model.device = device
    summary_writer = SummaryWriter(log_dir=logs_path) if arg_parser.log else None

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

            if arg_parser.optimizer:
                loss.backward()
                optimizer.step()
                optimizer.optimizer.zero_grad()
            else:
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
            if arg_parser.train_load:
                save_model(arg_parser.models_path,
                           f"{file_name_epoch_indep}_epoch_{epoch + arg_parser.epoch_to_load}.pt", model,
                           device)
            else:
                save_model(arg_parser.models_path,
                           f"{file_name_epoch_indep}_epoch_{epoch + arg_parser.epoch_to_load}.pt", model,
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

def format_lf(strings_model, string_gold):
    return [''.join(string_model.split(' ')) for string_model in strings_model], ''.join(string_gold.split(' '))


def format_lf(strings_model, string_gold):
    return [''.join(string_model.split(' ')) for string_model in strings_model], ''.join(string_gold.split(' '))


def jaccard_similarity(tokens1, tokens2):
    set1 = set([char for char in tokens1])
    set2 = set([char for char in tokens2])
    score = len(set1 & set2) / len(set1 | set2)
    return score


def jaccard(model_queries, gold_queries):
    n = len(model_queries)
    print(n)
    assert n == len(gold_queries)
    score = 0
    for i in tqdm(range(n)):
        score += jaccard_similarity(model_queries[i][0], gold_queries[i])
    return score / n


def jaccard_strict(model_queries, gold_queries):
    n = len(model_queries)
    print(n)
    assert n == len(gold_queries)
    score = 0
    for i in tqdm(range(n)):
        x = jaccard_similarity(tokens1=model_queries[i][0], tokens2=gold_queries[i])
        if x == 1:
            score += 1
    return score / n


def strict_evaluation(model_queries, gold_queries):
    n = len(model_queries)
    assert n == len(gold_queries)
    return sum([model_queries[x][0] == gold_queries[x] for x in range(n)]) / n


def knowledge_based_evaluation(model_queries, gold_queries, domain=domains.GeoqueryDomain()):
    '''
    Evaluate the model through knowledge-base metrics
    '''
    n_correct = 0
    n_wrong = 0
    score = 0
    for i, gold_target in tqdm(enumerate(gold_queries)):
        derivs, denotation_correct_list = domain.compare_answers(true_answers=[gold_target],
                                                                 all_derivs=[model_queries[i]])
        if derivs is not None:
            n_correct += 1
            score += int(denotation_correct_list[0])
            assert len(denotation_correct_list) == 1
        else:
            n_wrong += 1
    print(f"denotation matching fucked up {n_wrong / (n_correct + n_wrong):.2f} of the time, thks boloss")
    return score / n_correct


def decoding(loaded_model, test_dataset, arg_parser):
    beam_size = arg_parser.beam_size
    max_len = arg_parser.max_decode_len
    decoding_method = loaded_model.beam_search if arg_parser.decoding == 'beam_search' else loaded_model.decode_greedy
    loaded_model.eval()
    model_outputs = []
    model_outputs_kb = []
    gold_queries_kb = []
    gold_queries = []
    count_ = 0
    with torch.no_grad():
        for src_sent_batch, gold_target in tqdm(data_iterator(test_dataset, batch_size=1, shuffle=False), total=280):
            example_hyps = decoding_method(src_sent=src_sent_batch, max_len=max_len, beam_size=beam_size)
            strings_model = [detokenize(example_hyp) for example_hyp in example_hyps]
            string_gold = gold_target[0]
            model_outputs_kb.append(strings_model)
            gold_queries_kb.append(model_outputs)
            strings_model, string_gold = format_lf(strings_model, string_gold)
            model_outputs.append(strings_model)
            gold_queries.append(string_gold)
            count_ += 1
            if count_ > 5:
                break
    return model_outputs, gold_queries, model_outputs_kb, gold_queries_kb

def test(arg_parser):
    file_name_epoch_indep = get_model_name(arg_parser)
    recombination = arg_parser.recombination_method
    test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'test', 't280.tsv')#f"{recombination}_recomb.tsv")
    vocab = Vocab(f'bert-{arg_parser.BERT}-uncased')
    file_path = os.path.join(arg_parser.models_path, f"{file_name_epoch_indep}_epoch_{arg_parser.epoch_to_load}.pt")
    model_type = TSP if arg_parser.TSP_BSP else BSP
    model = model_type(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_int,
                       d_k=arg_parser.d_k, h=arg_parser.h, n_layers=arg_parser.n_layers,
                       dropout_rate=arg_parser.dropout, max_len_pe=arg_parser.max_len_pe, bert_name=arg_parser.BERT)

    load_model(file_path=file_path, model=model)
    evaluation_methods = {'strict': strict_evaluation, 'jaccard': jaccard, 'jaccard_strict': jaccard_strict}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.device = device
    parsing_outputs, gold_queries, parsing_kb, gold_kb = decoding(model, test_dataset, arg_parser)

    for eval_name, eval_method in evaluation_methods.items():
        if eval_method == 'Knowledge-based':
            test_accuracy = eval_method(parsing_kb, gold_kb)
        else:
            test_accuracy = eval_method(parsing_outputs, gold_queries)
        print(f"evaluation method is {eval_name}")
        print(
            f"test accuracy, model {eval_name}_{arg_parser.decoding}_{file_name_epoch_indep}_{arg_parser.epoch_to_load}: {test_accuracy:2f}")

    outfile = os.path.join(arg_parser.data_folder, os.path.join(arg_parser.out_folder,
                                                                f"{eval_name}_{arg_parser.decoding}_{file_name_epoch_indep}_{arg_parser.epoch_to_load}.txt"))
    with open(outfile, 'w') as f:
        for parsing_output in parsing_outputs:
            f.write(''.join(parsing_output) + '\n')
    return None


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
