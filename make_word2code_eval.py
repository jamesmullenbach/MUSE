"""
    Get top N words for set of 100 codes by all three methods, arrange them in evaluation markdown
"""
import argparse
from collections import defaultdict
import csv
import os

from gensim.models import KeyedVectors
from tqdm import tqdm

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

top100_words = [line.split()[0] for line in open('top_tfidf_words25.txt')][:100]
code2desc = {}
with open('/data/mimicdata/raw/ICD9_descriptions') as f:
    r = csv.reader(f, delimiter='\t')
    for row in r:
        code2desc[row[0]] = row[1]
code2desc[''] = ''

all_word2codes = defaultdict(set)
method2word2codes = {}
#methods = ['uwc', 'cui', 'both']
methods = ['both', 'unsup']
#dirs = [args.uwc_dir, args.cui_dir, args.both_dir]
dirs = [args.both_dir, args.unsup_dir]
for dr, method in zip(dirs, methods):
    model = KeyedVectors.load_word2vec_format(os.path.join(dr, 'vectors.txt'))

    word2codes = defaultdict(list)
    for word in tqdm(top100_words):
        code_list = [reformat(code) for code, score in model.most_similar(word, topn=len(model.wv.vocab)) if '_' in code and reformat(code) in code2desc][:N]
        #ignore words if other codes have them, to reduce annotation effort
        word2codes[word] = [code for code in code_list if code not in all_word2codes[word]]
        all_word2codes[word].update(set(word2codes[word]))

    method2word2codes[method] = word2codes

with open('evaluation_word2code25.md', 'w') as of:
    of.write("""### Instructions

    You will see a word accompanied by three lists of codes with their descriptions. Please circle all the codes you see that are clinically relevant to the given word. Treat any misspellings as if the term were properly spelled.

""") 
    for it, word in enumerate(top100_words):
        of.write(f"#### {word}\n")
        of.write("|A|B|\n")
        of.write("|-|-|\n")
        #list_uwc = method2code2words['uwc'][code]
        list_both = method2word2codes['both'][word]
        #list_cui = [word for word in method2code2words['cui'][code] if word not in list_uwc]
        #list_both = [word for word in method2code2words['both'][code] if word not in list_uwc and word not in list_cui]
        list_unsup = [code for code in method2word2codes['unsup'][word] if code not in list_both]
        #list_cui.extend([''] * (N - len(list_cui)))
        list_unsup.extend([''] * (N - len(list_unsup)))
        #for w1, w2, w3 in zip(list_uwc, list_cui, list_both):
        for c1, c2 in zip(list_both, list_unsup):
            if c1 not in code2desc or c2 not in code2desc:
                import pdb; pdb.set_trace()
            of.write(f"|{c1}: {code2desc[c1]}|{c2}: {code2desc[c2]}|\n")
        of.write("\n")
