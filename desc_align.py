"""
    Script to align code representations from word2vec and from combining (averaging) word embeddings of the description
"""
import argparse

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cosine

parser = argparse.ArgumentParser()
parser.add_argument('code_emb', type=str, help="path to code embeddings")
parser.add_argument('desc_emb', type=str, help="path to description embeddings")
parser.add_argument('eval_pairs', type=str, help="path to pairs for evaluation")
args = parser.parse_args()

#load embeddings as word2vec dicts
print("loading code embeddings")
f = open(args.code_emb)
code2vec = {}
next(f)
for row in f:
    row = row.rstrip()
    code = row.split()[0]
    vec = np.fromstring(row[len(code):], sep=' ')
    code2vec[code] = vec
    emb_dim = len(vec)
f.close()

print("loading desc embeddings")
f = open(args.desc_emb)
desc2vec = {}
next(f)
for row in f:
    row = row.rstrip()
    code = row.split()[0]
    vec = np.fromstring(row[len(code):], sep=' ')
    desc2vec[code] = vec

print("loading eval codes")
eval_codes = set()
with open(args.eval_pairs) as f:
    for row in f:
        eval_codes.add(row.split()[0])

#get common vocabulary
common = set()
for code in desc2vec.keys():
    #trim the '_w'
    code = code[:-2]
    if code in code2vec.keys() and code not in eval_codes:
        common.add(code)
vocab = sorted(common)

#make matrices
X, Y = [], []
for code in vocab:
    X.append(code2vec[code])
    Y.append(desc2vec[code + '_w'])
X = np.array(X).transpose()
Y = np.array(Y).transpose()

#do procrustes
if X.any() and Y.any():
    U, s, Vh = svd(Y @ X.transpose(), full_matrices=False)
    W = U @ Vh

#evaluate
ranks = []
rranks = []
for cde in sorted(eval_codes):
    desc_repr = W @ desc2vec[cde + '_w']
    code2sim = {}
    for code2, vec in code2vec.items():
        code2sim[code2] = cosine(desc_repr, vec)
    codes_ranked = np.array([code for code, score in sorted(code2sim.items(), key=lambda x: x[1])])
    rank = np.where(codes_ranked == cde)[0][0] + 1
    ranks.append(rank)
    rranks.append(1/rank)
print(np.mean(ranks))
print(np.mean(rranks))
