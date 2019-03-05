import torch
from argparse import ArgumentParser
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
parser.add_argument("--epoch_to_load", default=45, type=int) #TO BE CHECKED
parser.add_argument("--seed", default=1515, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--log_dir", default='logs', type=str)
parser.add_argument("--log", default=True, type=bool)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--save_every", default=5, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--decoding", default='beam_search', type=str)
parser.add_argument("--beam_size", default=5, type=int)
parser.add_argument("--max_decode_len", default=105, type=int)
parser.add_argument("--domain", default='geoquery', type=str)
parser.add_argument("--recombination", default='', type=str)


#TODO more proper management of different datasets and BSP / TSP split
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
    recombination = arg_parser.recombination
    if len(recombination) > 0:
        train_dataset = get_dataset_finish_by(arg_parser.data_folder, 'train', f"{recombination}_recomb.tsv")
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'dev', f"{recombination}_recomb.tsv")
        file_name_epoch_indep = f"TSP_recomb_{recombination}"
    else:
        train_dataset = get_dataset_finish_by(arg_parser.data_folder, 'train', '600.tsv')
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'dev', '100.tsv')
        file_name_epoch_indep = "TSP"
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
            save_model(arg_parser.models_path, f"{file_name_epoch_indep}_epoch_{epoch}.pt", model,
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


def test(arg_parser):
    recombination = arg_parser.recombination
    if len(recombination) > 0:
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'test', f"{recombination}_recomb.tsv")
        file_name_epoch_indep = f"TSP_recomb_{recombination}"
    else:
        test_dataset = get_dataset_finish_by(arg_parser.data_folder, 'test', '280.tsv')
        file_name_epoch_indep = "TSP"

    vocab = Vocab(arg_parser.BERT)
    file_path = os.path.join(arg_parser.models_path, f"{file_name_epoch_indep}_epoch_{arg_parser.epoch_to_load}.pt")
    model = TSP(input_vocab=vocab, target_vocab=vocab, d_model=arg_parser.d_model, d_int=arg_parser.d_model,
                n_layers=arg_parser.n_layers, dropout_rate=arg_parser.dropout)
    load_model(file_path=file_path, model=model)
    evaluation_methods = {'strict': strict_evaluation, 'jaccard': jaccard, 'jaccard_strict': jaccard_strict}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    parsing_outputs, gold_queries = decoding(model, test_dataset, arg_parser)

    for eval_name, eval_method in evaluation_methods.items():
        test_accuracy = eval_method(parsing_outputs, gold_queries)
        print(f"evaluation method is {eval_name}")
        print(
            f"test accuracy, model {eval_name}_{arg_parser.decoding}_TSP_{arg_parser.epoch_to_load}: {test_accuracy:2f}")

    outfile = os.path.join(arg_parser.data_folder, os.path.join(arg_parser.out_folder,
                                                                f"{eval_name}_{arg_parser.decoding}_TSP_{arg_parser.epoch_to_load}.txt"))
    with open(outfile, 'w') as f:
        for parsing_output in parsing_outputs:
            f.write(''.join(parsing_output) + '\n')
    return None


def decoding(loaded_model, test_dataset, arg_parser):
    beam_size = arg_parser.beam_size
    max_len = arg_parser.max_decode_len
    decoding_method = loaded_model.beam_search if arg_parser.decoding == 'beam_search' else loaded_model.decode_greedy
    loaded_model.eval()
    model_outputs = []
    gold_queries = []
    with torch.no_grad():
        for src_sent_batch, gold_target in tqdm(data_iterator(test_dataset, batch_size=1, shuffle=False), total=280):
            example_hyps = decoding_method(src_sent=src_sent_batch, max_len=max_len, beam_size=beam_size)
            model_outputs.append(example_hyps[0])
            gold_queries.append(loaded_model.target_vocab.to_input_tokens(gold_target)[0])
    return model_outputs, gold_queries


def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    score = len(set1 & set2) / len(set1 | set2)
    return score


def jaccard(model_queries, gold_queries):
    n = len(model_queries)
    print(n)
    assert n == len(gold_queries)
    score = 0
    for i in tqdm(range(n)):
        score += jaccard_similarity(model_queries[i], gold_queries[i])
    return score / n


def jaccard_strict(model_queries, gold_queries):
    n = len(model_queries)
    print(n)
    assert n == len(gold_queries)
    score = 0
    for i in tqdm(range(n)):
        x = jaccard_similarity(model_queries[i], gold_queries[i])
        if x == 1:
            score += 1
    return score / n


#
# def knowledge_based_evaluation(model_queries, gold_queries, domain = domains.GeoqueryDomain()):
#     '''
#     Evaluate the model through knowledge-base metrics
#     '''
#     is_correct_list = []
#     tokens_correct_list = []
#     x_len_list = []
#     y_len_list = []
#
#     #print("This is the predicted queries \n", model_queries[0][0])
#     #print("This is the expected queries \n", gold_queries)
#
#     if domain:
#     #    all_derivs = [decoding_method(sources=ex, max_len=max_len, beam_size=beam_size) for ex in dataset]
#     #    true_answers = [ex for ex in dataset]
#         derivs, denotation_correct_list = domain.compare_answers(model_queries, gold_queries)
#     #else:
#     #    derivs = [decoding_method(model, ex)[0] for ex in dataset]
#     #    denotation_correct_list = None
#
#     for i, ex in enumerate(dataset):
#         print('Example %d' % i)
#         print('  x      = "%s"' % ex.x_str)
#         print('  y      = "%s"' % ex.y_str)
#         prob = derivs[i].p
#         y_pred_toks = derivs[i].y_toks
#         y_pred_str = ' '.join(y_pred_toks)
#
#     # Compute accuracy metrics
#         is_correct = (y_pred_str == ex.y_str)
#         tokens_correct = sum(a == b for a, b in zip(y_pred_toks, ex.y_toks))
#         is_correct_list.append(is_correct)
#         tokens_correct_list.append(tokens_correct)
#         x_len_list.append(len(ex.x_toks))
#         y_len_list.append(len(ex.y_toks))
#         print('  y_pred = "%s"' % y_pred_str)
#         print('  sequence correct = %s' % is_correct)
#         print('  token accuracy = %d/%d = %g' % (tokens_correct, len(ex.y_toks), float(tokens_correct) / len(ex.y_toks)))
#         if denotation_correct_list:
#             denotation_correct = denotation_correct_list[i]
#             print('  denotation correct = %s' % denotation_correct)
#     #print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
#     #                     x_len_list, y_len_list, denotation_correct_list)

def strict_evaluation(model_queries, gold_queries):
    n = len(model_queries)
    assert n == len(gold_queries)
    return sum([model_queries[x] == gold_queries[x] for x in range(n)]) / n


'''
def load_dataset(filename, domain):
  dataset = []
  with open(filename) as f:
    for line in f:
      x, y = line.rstrip('\n').split('\t')
      if domain:
        y = domain.preprocess_lf(y)
      dataset.append((x, y))
  return dataset

def get_input_vocabulary(dataset):
  sentences = [x[0] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff)

def get_output_vocabulary(dataset):
  sentences = [x[1] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.output_embedding_dim)


def preprocess_data(model, raw):
  in_vocabulary = model.in_vocabulary
  out_vocabulary = model.out_vocabulary
  lexicon = model.lexicon

  data = []
  for raw_ex in raw:
    x_str, y_str = raw_ex
    ex = Example(x_str, y_str, in_vocabulary, out_vocabulary, lexicon,
                 reverse_input=OPTIONS.reverse_input)
    data.append(ex)
  return data
'''

if __name__ == '__main__':
    args = parser.parse_args()
    # train(args)
    test(args)
