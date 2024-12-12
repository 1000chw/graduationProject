# Apache License v2.0

# Tong Guo
# Sep30, 2019


import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets
from data_and_model.output_entity import *

# BERT
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, AutoConfig, AutoTokenizer, AutoModel

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CORENLP_HOME'] = "C:\\Users\\1000c\\Desktop\\stanford-corenlp-4.5.7"


def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=True)
    parser.add_argument('--do_infer', default=True)
    parser.add_argument('--infer_loop', default=False)

    parser.add_argument("--trained", default=True)

    parser.add_argument('--fine_tune',
                        default=True,
                        help="If present, BERT is trained.")

    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=8, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=512, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', default=True, help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


def get_roberta(RoBERTa_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(RoBERTa_PT_PATH, f'berta_config_{bert_type}.json')
    vocab_file = os.path.join(RoBERTa_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(RoBERTa_PT_PATH, f'pytorch_model_{bert_type}.bin')

    roberta_config = AutoConfig.from_pretrained('klue/roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    en_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    tokenizer.add_tokens(list(en_tokenizer.vocab.keys()))

    model_roberta = AutoModel.from_pretrained('klue/roberta-base')
    model_roberta.resize_token_embeddings(len(tokenizer.vocab))


    if no_pretraining:
        pass
    else:
        model_roberta.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_roberta.to(device)

    return model_roberta, tokenizer, roberta_config


def get_models(args, RoBERTa_PT_PATH, trained=False, path_model_roberta=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_roberta, tokenizer, bert_config = get_roberta(RoBERTa_PT_PATH, args.bert_type, args.do_lower_case,
                                                        args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_roberta != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_roberta, weights_only=True)
        else:
            res = torch.load(path_model_roberta, map_location='cpu', weights_only=True)
        model_roberta.load_state_dict(res['model_roberta'])
        model_roberta.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model, weights_only=True)
        else:
            res = torch.load(path_model, map_location='cpu', weights_only=False)

        model.load_state_dict(res['model'])

    return model, model_roberta, tokenizer, bert_config


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


BERT_PT_PATH = './'
path_model_roberta = './model_roberta_best.pt'
path_model = './model_best.pt'
parser = argparse.ArgumentParser()
args = construct_hyper_param(parser)
model, model_roberta, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                       path_model_roberta=path_model_roberta, path_model=path_model)


def infer(nlu1,
          table_name, data_table, path_db, db_name,
          model, model_roberta, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_roberta.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # Get inputs
    nlu = [nlu1]
    # nlu_t1 = tokenize_corenlp(client, nlu1)
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]

    tb1 = data_table[0]
    for table in data_table:
        if table["id"] == table_name:
            tb1 = table
            break
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_roberta(bert_config, model_roberta, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size)
    # sort and generate
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])
    pr_sql_q = [pr_sql_q1]

    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    if show_answer_only:
        print(f'Q: {nlu[0]}')
        print(f'A: {pr_ans[0]}')
        print(f'SQL: {pr_sql_q}')

    else:
        print(f'START ============================================================= ')
        print(f'{hds}')
        if show_table:
            print(engine.show_table(table_name))
        print(f'nlu: {nlu}')
        print(f'pr_sql_i : {pr_sql_i}')
        print(f'pr_sql_q : {pr_sql_q}')
        print(f'pr_ans: {pr_ans}')
        print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans


import corenlp

client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

nlu1 = "Order Year 1998 의 Manufacturer는 누구인가요?"
path_db = './data_and_model'
db_name = 'train'
data_table = load_jsonl('./data_and_model/train.tables.jsonl')
table_name = '1-10007452-3'
n_Q = 1 #00000 if args.infer_loop else 1
for i in range(n_Q):
    if n_Q > 1:
        nlu1 = input('Type question: ')
    pr_sql_i, pr_ans = infer(
        nlu1,
        table_name, data_table, path_db, db_name,
        model, model_roberta, bert_config, max_seq_length=args.max_seq_length,
        num_target_layers=args.num_target_layers,
        beam_size=1, show_table=False, show_answer_only=False
    )