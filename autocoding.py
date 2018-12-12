"""
    Use aligned code and word embeddings for auto coding on MIMIC-III 50-label test set
"""

import argparse
import csv
import os

from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

from downstream_eval import all_metrics, print_metrics

parser = argparse.ArgumentParser()
parser.add_argument("word_emb", type=str, help="path to word embeddings")
parser.add_argument("code_emb", type=str, help="path to code embeddings")
parser.add_argument("test_data", type=str, help="path to test data")
parser.add_argument("diags", type=str, help="path to top codes for evaluation")
parser.add_argument("word_agg", choices=['max', 'mean'], help="method to aggregate word embeddings of the document (max or mean)")
args = parser.parse_args()

assert os.path.isfile(args.word_emb)
assert os.path.isfile(args.code_emb)
assert os.path.isfile(args.test_data)

word_emb = KeyedVectors.load_word2vec_format(args.word_emb)
code_emb = KeyedVectors.load_word2vec_format(args.code_emb)

#first get the set of codes for evaluation
eval_codes = sorted(['d_' + line.strip().replace('.', '') for line in open(args.diags)])
id2code = eval_codes
code2id = {code: i for i, code in enumerate(id2code)}

with open(args.test_data) as f:
    r = csv.reader(f)
    #header
    next(r)
    yhats = []
    ys = []
    yhat_raw = []
    for row in tqdm(r):
        text = row[2].split()
        codes = row[3].split(';')
        #convert code name format
        codes = ['d_' + code.replace('.', '') for code in codes]
        if args.word_agg == 'mean':
            doc_repr = np.mean([word_emb[word] for word in text if word in word_emb], 0)
        elif args.word_agg == 'max':
            doc_repr = np.max([word_emb[word] for word in text if word in word_emb], 0)
        code_dists = code_emb.wv.distances(doc_repr, other_words=eval_codes)
        #take the top 5 as true, and use 5th as threshold to create binary predictions
        thresh = sorted(code_dists, reverse=True)[4]
        yhat = code_dists >= thresh
        y = np.zeros(len(yhat))
        yhat_raw.append(code_dists)
        for code in codes:
            y[code2id[code]] = 1
        yhats.append(yhat)
        ys.append(y)
    yhats = np.array(yhats)
    ys = np.array(ys)
    metrics = all_metrics(yhats, ys, k=5, yhat_raw=yhat_raw, calc_auc=False)
    print_metrics(metrics)
