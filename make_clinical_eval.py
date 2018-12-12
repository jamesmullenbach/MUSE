"""
    Get top N words for set of 100 codes by all three methods, arrange them in evaluation markdown
"""
import argparse
from collections import defaultdict
import csv
import os

from gensim.models import KeyedVectors

N = 5

def reformat(code):
    #crop the 'd_'
    code = code[2:]
    if code.startswith('E'):
        if len(code) > 4:
            return code[:4] + '.' + code[4:]
        else:
            return code
    else:
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code

parser = argparse.ArgumentParser()
#parser.add_argument("uwc_dir", type=str, help="path to UWC alignment result dir")
#parser.add_argument("cui_dir", type=str, help="path to CUI alignment result dir")
parser.add_argument("both_dir", type=str, help="path to CUI+UWC alignment result dir")
parser.add_argument("unsup_dir", type=str, help="path to unsupervised alignment result dir")
args = parser.parse_args()

top100_codes = [line.split()[0] for line in open('../../data/fc_diag_all_vocab')][:100]
code2desc = {}
with open('/data/mimicdata/raw/ICD9_descriptions') as f:
    r = csv.reader(f, delimiter='\t')
    for row in r:
        codename = 'd_' + row[0].replace('.', '')
        code2desc[codename] = row[1]

all_code2words = defaultdict(set)
method2code2words = {}
#methods = ['uwc', 'cui', 'both']
methods = ['both', 'unsup']
#dirs = [args.uwc_dir, args.cui_dir, args.both_dir]
dirs = [args.both_dir, args.unsup_dir]
for dr, method in zip(dirs, methods):
    model = KeyedVectors.load_word2vec_format(os.path.join(dr, 'vectors.txt'))

    code2words = defaultdict(list)
    for code in top100_codes:
        word_list = [word for word, score in model.most_similar(code, topn=1000) if '_' not in word][:N]
        #ignore words if other codes have them, to reduce annotation effort
        code2words[code] = [word for word in word_list if word not in all_code2words[code]]
        all_code2words[code].update(set(code2words[code]))

    method2code2words[method] = code2words

with open('evaluation2.md', 'w') as of:
    of.write("""### Instructions

    You will see an ICD-9 code with its description, accompanied by three lists of individual words. Please circle all the words you see that are clinically relevant to the given code. Treat any misspellings as if the term were properly spelled.

""") 
    for code in top100_codes:
        code_str = reformat(code)
        of.write(f"#### {code_str}: {code2desc[code]}\n")
        of.write("|A|B|\n")
        of.write("|-|-|\n")
        #list_uwc = method2code2words['uwc'][code]
        list_both = method2code2words['both'][code]
        #list_cui = [word for word in method2code2words['cui'][code] if word not in list_uwc]
        #list_both = [word for word in method2code2words['both'][code] if word not in list_uwc and word not in list_cui]
        list_unsup = [word for word in method2code2words['unsup'][code] if word not in list_both]
        #list_cui.extend([''] * (N - len(list_cui)))
        list_unsup.extend([''] * (N - len(list_unsup)))
        #for w1, w2, w3 in zip(list_uwc, list_cui, list_both):
        for w1, w2 in zip(list_both, list_unsup):
            of.write(f"|{w1}|{w2}|\n")
        of.write("\n")
