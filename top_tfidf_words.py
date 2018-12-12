from collections import defaultdict

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.downloader as api
import numpy as np
from tqdm import tqdm

vocab = [line.split()[0] for line in open('../../data/NOTEEVENTS_proc_20_5_128.embed').readlines()[1:]]
ix2word = sorted(vocab)
word2ix = {word:ix for ix,word in enumerate(ix2word)}

def doc2bow(doc):
    bow = [0] * len(vocab)
    for word in doc:
        if word in word2ix:
            bow[word2ix[word]] += 1
    return [(ix, cnt) for ix, cnt in enumerate(bow) if cnt > 0]
    
print("loading dataset...")
dataset = [doc.split(',')[10].split() for doc in open('../../data/disch_proc.csv').readlines()]
print("done")
corpus = []
for line in tqdm(dataset):
    corpus.append(doc2bow(line))
    
model = TfidfModel(corpus)

ix2vals = defaultdict(list)
for bow in corpus:
    for (ix, cnt) in bow:
        ix2vals[ix].append(cnt)
        
word2avgtfidf = {}
for ix, vals in ix2vals.items():
    if len(vals) >= 300:
        word2avgtfidf[ix2word[ix]] = np.mean([val/len(vals) for val in vals])
    
words = [word for word, score in sorted(word2avgtfidf.items(), key=lambda x: x[1], reverse=True)[:100]]
with open('top_tfidf_words300.txt', 'w') as of:
    for word in words:
        of.write(word + '\n')
        
