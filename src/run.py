import torch
from argparse import ArgumentParser
import os
from utils import read_GeoQuery, data_iterator
from pytorch_pretrained_bert.modeling import BertModel
from tokens_vocab import Vocab
# from semantic_parser import BSP

parser = ArgumentParser()
parser.add_argument("--data_folder", type=str, default="geoQueryData")
parser.add_argument("--data_file", type=str, default="geo880_train100.tsv")
parser.add_argument("--BERT", default="bert-base-uncased", type=str, required=True,
                    help="bert-base-uncased, bert-large-uncased")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--valid_iters", default=2000, type=int)
parser.add_argument("--save_path", default='models', type=str)
parser.add_argument("--log_every", default=10, type=int)




def get_file_paths(directory):
    files_list = [os.path.join(directory, f) for f in os.listdir(directory) if
                  os.path.isfile(os.path.join(directory, f))]
    return files_list


def main(arg_parser):
    test_dir = os.path.join(arg_parser.data_folder, 'test')
    train_dir = os.path.join(arg_parser.data_folder, 'train')
    dev_dir = os.path.join(arg_parser.data_folder, 'dev')

    test_file_paths = get_file_paths(test_dir)
    vocab = Vocab(arg_parser.BERT)
    import pdb;
    pdb.set_trace()
    # BERT_encoder = BertModel.from_pretrained(arg_parser.BERT,
    #                                          cache_dir=os.path.join('BERT_pretrained_models', arg_parser.BERT))
    # dataset = read_GeoQuery(file_paths=test_file_paths)
    # with torch.no_grad():
    #     for i, (input_sentences_batch, target_queries_batch) in enumerate(
    #             data_iterator(dataset, batch_size=arg_parser.batch_size, shuffle=True)):
    #         input_tensor = vocab.to_input_tensor(input_sentences_batch, device='cpu')
    #         output_layers, _ = BERT_encoder(input_tensor, output_all_encoded_layers=False)
    #         # output_layers of dim bsize, max_len, 768 or 1024 (base or large)


# def train(arg_parser):
#     """ Train the NMT Model.
#     @param args (Dict): args from cmd line
#     """
#     train_dir = os.path.join(arg_parser.data_folder, 'train')
#     train_data = read_GeoQuery(file_paths=get_file_paths(train_dir))
#
#     dev_dir = os.path.join(arg_parser.data_folder, 'dev')
#     dev_data = read_GeoQuery(file_paths=get_file_paths(dev_dir))
#
#     test_dir = os.path.join(arg_parser.data_folder, 'test')
#     test_data = read_GeoQuery(file_paths=get_file_paths(test_dir))
#
#     batch_size = arg_parser.batch_size
#
#     #TODO adapt default parameters to our problem
#     clip_grad = arg_parser.clip_grad
#     valid_niter = arg_parser.valid_iters
#     log_every = arg_parser.log_every
#     model_save_path = arg_parser.save_path
#
#     vocab = Vocab(arg_parser.BERT)
#
#     model = NMT(embed_size=int(args['--embed-size']),
#                 hidden_size=int(args['--hidden-size']),
#                 dropout_rate=float(args['--dropout']),
#                 vocab=vocab, no_char_decoder=args['--no-char-decoder'])
#     model.train()
#
#     uniform_init = float(args['--uniform-init'])
#     if np.abs(uniform_init) > 0.:
#         print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
#         for p in model.parameters():
#             p.data.uniform_(-uniform_init, uniform_init)
#
#     vocab_mask = torch.ones(len(vocab.tgt))
#     vocab_mask[vocab.tgt['<pad>']] = 0
#
#     device = torch.device("cuda:0" if args['--cuda'] else "cpu")
#     print('use device: %s' % device, file=sys.stderr)
#
#     model = model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
#
#     num_trial = 0
#     train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
#     cum_examples = report_examples = epoch = valid_num = 0
#     hist_valid_scores = []
#     train_time = begin_time = time.time()
#     print('begin Maximum Likelihood training')
#
#     while True:
#         epoch += 1
#
#         for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
#             train_iter += 1
#
#             optimizer.zero_grad()
#
#             batch_size = len(src_sents)
#
#             example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
#             batch_loss = example_losses.sum()
#             loss = batch_loss / batch_size
#
#             loss.backward()
#
#             # clip gradient
#             grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#
#             optimizer.step()
#
#             batch_losses_val = batch_loss.item()
#             report_loss += batch_losses_val
#             cum_loss += batch_losses_val
#
#             tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
#             report_tgt_words += tgt_words_num_to_predict
#             cum_tgt_words += tgt_words_num_to_predict
#             report_examples += batch_size
#             cum_examples += batch_size
#
#             if train_iter % log_every == 0:
#                 print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
#                       'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
#                                                                                          report_loss / report_examples,
#                                                                                          math.exp(
#                                                                                              report_loss / report_tgt_words),
#                                                                                          cum_examples,
#                                                                                          report_tgt_words / (
#                                                                                                      time.time() - train_time),
#                                                                                          time.time() - begin_time),
#                       file=sys.stderr)
#
#                 train_time = time.time()
#                 report_loss = report_tgt_words = report_examples = 0.
#
#             # perform validation
#             if train_iter % valid_niter == 0:
#                 print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
#                                                                                              cum_loss / cum_examples,
#                                                                                              np.exp(
#                                                                                                  cum_loss / cum_tgt_words),
#                                                                                              cum_examples),
#                       file=sys.stderr)
#
#                 cum_loss = cum_examples = cum_tgt_words = 0.
#                 valid_num += 1
#
#                 print('begin validation ...', file=sys.stderr)
#
#                 # compute dev. ppl and bleu
#                 dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
#                 valid_metric = -dev_ppl
#
#                 print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
#
#                 is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
#                 hist_valid_scores.append(valid_metric)
#
#                 if is_better:
#                     patience = 0
#                     print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
#                     model.save(model_save_path)
#
#                     # also save the optimizers' state
#                     torch.save(optimizer.state_dict(), model_save_path + '.optim')
#                 elif patience < int(args['--patience']):
#                     patience += 1
#                     print('hit patience %d' % patience, file=sys.stderr)
#
#                     if patience == int(args['--patience']):
#                         num_trial += 1
#                         print('hit #%d trial' % num_trial, file=sys.stderr)
#                         if num_trial == int(args['--max-num-trial']):
#                             print('early stop!', file=sys.stderr)
#                             exit(0)
#
#                         # decay lr, and restore from previously best checkpoint
#                         lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
#                         print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
#
#                         # load model
#                         params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
#                         model.load_state_dict(params['state_dict'])
#                         model = model.to(device)
#
#                         print('restore parameters of the optimizers', file=sys.stderr)
#                         optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
#
#                         # set new lr
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] = lr
#
#                         # reset patience
#                         patience = 0
#
#             if epoch == int(args['--max-epoch']):
#                 print('reached maximum number of epochs!', file=sys.stderr)
#                 exit(0)
#

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
