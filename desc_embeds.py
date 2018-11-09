# coding: utf-8
import csv
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

model = KeyedVectors.load_word2vec_format('../../data/NOTEEVENTS_proc_20_5_128.embed')
codes = [line.split()[0] for line in open('../../data/fc_full_all_trained_30_128_10').readlines()[1:]]
       
codes = set(codes)
code2repr = {}
print("reading...")
with open('../../data/D_ICD_DIAGNOSES.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in tqdm(r):
        code = 'd_' + row[1]
        desc = [tok.lower() for tok in row[-1].split() if not tok.isnumeric() and tok.lower() in model]
        desc_repr = np.mean([model[word] for word in desc],0)
        if code in set(codes) and not np.any(np.isnan(desc_repr)):
            code2repr[code] = desc_repr
        
diag_codes = set([code for code in codes if code.startswith('d')])
print("writing...")
with open('descs.txt', 'w') as of:
    of.write(f'{len(code2repr)} 128\n')
    for code in sorted(code2repr.keys()):
        of.write(code + '_w' + ' ' + ' '.join([str(x) for x in code2repr[code]]) + '\n')
