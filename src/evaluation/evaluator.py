# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
from logging import getLogger
from copy import deepcopy
import os
from collections import defaultdict

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor

from . import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from . import get_word_translation_accuracy
from . import load_europarl_data, get_sent_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary
from src.utils import get_idf


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.glo_emb = trainer.glo_emb

    def monolingual_wordsim(self, to_log):
        """
        Evaluation on monolingual word similarity.
        """
        src_ws_scores = get_wordsim_scores(
            self.src_dico.lang, self.src_dico.word2id,
            self.mapping(self.src_emb.weight).data.cpu().numpy()
        )
        tgt_ws_scores = get_wordsim_scores(
            self.tgt_dico.lang, self.tgt_dico.word2id,
            self.tgt_emb.weight.data.cpu().numpy()
        ) if self.params.tgt_lang else None
        if src_ws_scores is not None:
            src_ws_monolingual_scores = np.mean(list(src_ws_scores.values()))
            logger.info("Monolingual source word similarity score average: %.5f" % src_ws_monolingual_scores)
            to_log['src_ws_monolingual_scores'] = src_ws_monolingual_scores
            to_log.update({'src_' + k: v for k, v in src_ws_scores.items()})
        if tgt_ws_scores is not None:
            tgt_ws_monolingual_scores = np.mean(list(tgt_ws_scores.values()))
            logger.info("Monolingual target word similarity score average: %.5f" % tgt_ws_monolingual_scores)
            to_log['tgt_ws_monolingual_scores'] = tgt_ws_monolingual_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_ws_scores.items()})
        if src_ws_scores is not None and tgt_ws_scores is not None:
            ws_monolingual_scores = (src_ws_monolingual_scores + tgt_ws_monolingual_scores) / 2
            logger.info("Monolingual word similarity score average: %.5f" % ws_monolingual_scores)
            to_log['ws_monolingual_scores'] = ws_monolingual_scores

    def monolingual_wordanalogy(self, to_log):
        """
        Evaluation on monolingual word analogy.
        """
        src_analogy_scores = get_wordanalogy_scores(
            self.src_dico.lang, self.src_dico.word2id,
            self.mapping(self.src_emb.weight).data.cpu().numpy()
        )
        if self.params.tgt_lang:
            tgt_analogy_scores = get_wordanalogy_scores(
                self.tgt_dico.lang, self.tgt_dico.word2id,
                self.tgt_emb.weight.data.cpu().numpy()
            )
        if src_analogy_scores is not None:
            src_analogy_monolingual_scores = np.mean(list(src_analogy_scores.values()))
            logger.info("Monolingual source word analogy score average: %.5f" % src_analogy_monolingual_scores)
            to_log['src_analogy_monolingual_scores'] = src_analogy_monolingual_scores
            to_log.update({'src_' + k: v for k, v in src_analogy_scores.items()})
        if self.params.tgt_lang and tgt_analogy_scores is not None:
            tgt_analogy_monolingual_scores = np.mean(list(tgt_analogy_scores.values()))
            logger.info("Monolingual target word analogy score average: %.5f" % tgt_analogy_monolingual_scores)
            to_log['tgt_analogy_monolingual_scores'] = tgt_analogy_monolingual_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_analogy_scores.items()})

    def crosslingual_wordsim(self, to_log):
        """
        Evaluation on cross-lingual word similarity.
        """
        src_emb = self.mapping(self.src_emb.weight).data.cpu().numpy()
        tgt_emb = self.tgt_emb.weight.data.cpu().numpy()
        # cross-lingual wordsim evaluation
        src_tgt_ws_scores = get_crosslingual_wordsim_scores(
            self.src_dico.lang, self.src_dico.word2id, src_emb,
            self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
        )
        if src_tgt_ws_scores is None:
            return
        ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
        logger.info("Cross-lingual word similarity score average: %.5f" % ws_crosslingual_scores)
        to_log['ws_crosslingual_scores'] = ws_crosslingual_scores
        to_log.update({'src_tgt_' + k: v for k, v in src_tgt_ws_scores.items()})

    def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                self.src_dico.lang, self.src_dico.word2id, src_emb,
                self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
                method=method,
                dico_eval=self.params.dico_eval
            )
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])

    def sent_translation(self, to_log):
        """
        Evaluation on sentence translation.
        Only available on Europarl, for en - {de, es, fr, it} language pairs.
        """
        lg1 = self.src_dico.lang
        lg2 = self.tgt_dico.lang

        # parameters
        n_keys = 200000
        n_queries = 2000
        n_idf = 300000

        # load europarl data
        if not hasattr(self, 'europarl_data'):
            self.europarl_data = load_europarl_data(
                lg1, lg2, n_max=(n_keys + 2 * n_idf)
            )

        # if no Europarl data for this language pair
        if not self.europarl_data:
            return

        # mapped word embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        # get idf weights
        idf = get_idf(self.europarl_data, lg1, lg2, n_idf=n_idf)

        for method in ['nn', 'csls_knn_10']:

            # source <- target sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.src_dico.lang, self.src_dico.word2id, src_emb,
                self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

            # target <- source sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
                self.src_dico.lang, self.src_dico.word2id, src_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])

    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
        for dico_method in ['nn', 'csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 10000
            # temp params / dictionary generation
            _params = deepcopy(self.params)
            _params.dico_method = dico_method
            _params.dico_build = dico_build
            _params.dico_threshold = 0
            _params.dico_max_rank = 10000
            _params.dico_min_size = 0
            _params.dico_max_size = dico_max_size
            s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
            t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
            dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates, log=False)
            # mean cosine
            if dico is None:
                mean_cosine = -1e9
            else:
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
            mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
            to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.monolingual_wordsim(to_log)
        self.crosslingual_wordsim(to_log)
        self.word_translation(to_log)
        self.sent_translation(to_log)
        self.dist_mean_cosine(to_log)

    def global_ranking_eval(self, to_log):
        assert self.glo_emb is not None
        src_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.src_lang))
        tgt_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.tgt_lang))
        print(len(src_emb.wv.vocab), len(tgt_emb.wv.vocab))
        assert len(src_emb.wv.vocab) == len(tgt_emb.wv.vocab)

        removed_keys1_file = self.params.removed_keys_file.replace('_0_', '_1_')
        missing_keys0 = set([line.strip() for line in open(self.params.removed_keys_file)])
        missing_keys1 = set([line.strip() for line in open(removed_keys1_file)])
        ranks_0 = []
        ranks_1 = []
        for key in missing_keys0:
            nn = self.glo_emb.wv.most_similar(key)[0][0]
            if nn in src_emb.wv.vocab:
                rank = src_emb.wv.rank(key, nn)
                ranks_0.append(rank)
        for key in missing_keys1:
            nn = self.glo_emb.wv.most_similar(key)[0][0]
            if nn in tgt_emb.wv.vocab:
                rank = tgt_emb.wv.rank(key, nn)
                ranks_1.append(rank)

        mean_rank = (np.mean(ranks_0) + np.mean(ranks_1)) / 2
        std_rank = np.std(np.concatenate((ranks_0, ranks_1)))
        print(f"mean rank of missing codes: {mean_rank} +/- {std_rank}")

    def desc_to_code_retrieval_eval(self, to_log, which_is_codes='src'):
        #load aligned embeddings
        src_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.src_lang))
        tgt_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.tgt_lang))
        #boilerplate variable assignment
        if which_is_codes == 'src':
            code2ix = {code:ix for ix,code in enumerate(sorted(src_emb.wv.vocab.keys()))}
            code_emb = src_emb
            word2ix = {word:ix for ix,word in enumerate(sorted(tgt_emb.wv.vocab.keys()))}
            word_emb = tgt_emb
        else:
            word2ix = {word:ix for ix,word in enumerate(sorted(src_emb.wv.vocab.keys()))}
            word_emb = src_emb
            code2ix = {code:ix for ix,code in enumerate(sorted(tgt_emb.wv.vocab.keys()))}
            code_emb = tgt_emb

        #load code2desc lookup
        code2desc = {}
        with open('../../data/D_ICD_DIAGNOSES.csv') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                code = 'd_' + row[1]
                desc = [tok.lower() for tok in word_tokenize(row[-1]) if not tok.isnumeric()]
                if code in code2ix:
                    code2desc[code] = desc
        ranks = []
        rank_1s = []
        #now go thru codes. get their description
        for cde, desc in code2desc.items():
            desc_repr = np.mean([word_emb[word] for word in desc if word in word2ix],0)
            if np.any(np.isnan(desc_repr)):
                continue
            #get similarity of each code to the description
            code_dists = code_emb.distances(desc_repr)
            closest = np.argsort(code_dists)
            rank = np.where(closest == code2ix[cde])[0][0]+1
            ranks.append(rank)
            if len(desc) == 1:
                rank_1s.append(rank)
        rranks = [1/rank for rank in ranks]
        rrank_1s = [1/rank for rank in rank_1s]

        mr = np.mean(ranks)
        mrstd = np.std(ranks)
        mrr = np.mean(rranks)
        mrrstd = np.std(rranks)

        mr1 = np.mean(rank_1s)
        mr1std = np.std(rank_1s)
        mrr1 = np.mean(rrank_1s)
        mrr1std = np.std(rrank_1s)
        print(f"mean rank: {mr} +/- {mrstd}")
        print(f"mean one-word description rank: {mr1} +/- {mr1std}")
        print(f"mrr: {mrr} +/- {mrrstd}")
        print(f"one-word description mrr: {mrr1} +/- {mrr1std}")
        return mr, mr1

    def word_to_codes_retrieval_eval(self, to_log, which_is_codes='src'):
        #load aligned embeddings
        src_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.src_lang))
        tgt_emb = KeyedVectors.load_word2vec_format(os.path.join(self.params.exp_path, 'vectors-%s.txt' % self.params.tgt_lang))
        #boilerplate variable assignment
        if which_is_codes == 'src':
            code2ix = {code:ix for ix,code in enumerate(sorted(src_emb.wv.vocab.keys()))}
            code_emb = src_emb
            word2ix = {word:ix for ix,word in enumerate(sorted(tgt_emb.wv.vocab.keys()))}
            word_emb = tgt_emb
        else:
            word2ix = {word:ix for ix,word in enumerate(sorted(src_emb.wv.vocab.keys()))}
            word_emb = src_emb
            code2ix = {code:ix for ix,code in enumerate(sorted(tgt_emb.wv.vocab.keys()))}
            code_emb = tgt_emb

        #load word2codes lookup
        word2codes = defaultdict(set)
        word_1s = set()
        single_codes = set()
        with open('../../data/D_ICD_DIAGNOSES.csv') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                cde = 'd_' + row[1]
                desc = word_tokenize(row[-1])
                for tok in desc:
                    tok = tok.lower()
                    if not tok.isnumeric():
                        if tok in word2ix and cde in code2ix:
                            word2codes[tok].add(cde)
                if len(desc) == 1 and desc[0].lower() in word2ix:
                    word_1s.add(desc[0].lower())
                    if cde in code2ix:
                        single_codes.add((desc[0].lower(), cde))
        ranks = []
        rranks = []
        rank_1s = []
        rrank_1s = []
        for word, codes in word2codes.items():
            code_dists = code_emb.distances(word_emb[word])
            closest = np.argsort(code_dists)
            wranks = []
            wrranks = []
            for cde in codes:
                rank = np.where(closest == code2ix[cde])[0][0]+1
                wranks.append(rank)
                wrranks.append(1/rank)
            mr = np.mean(wranks)
            mrr = np.mean(wrranks)
            ranks.append(mr)
            rranks.append(mrr)
            if word in word_1s:
                rank_1s.append(mr)
                rrank_1s.append(mrr)

        mr = np.mean(ranks)
        mrstd = np.std(ranks)
        mrr = np.mean(rranks)
        mrrstd = np.std(rranks)

        mr1 = np.mean(rank_1s)
        mr1std = np.std(rank_1s)
        mrr1 = np.mean(rrank_1s)
        mrr1std = np.std(rrank_1s)
        print(f"mean rank: {mr} +/- {mrstd}")
        print(f"mean one-word description rank: {mr1} +/- {mr1std}")
        print(f"mrr: {mrr} +/- {mrrstd}")
        print(f"one-word description mrr: {mrr1} +/- {mrr1std}")
        with open('single_word_codes.txt', 'w') as of:
            w = csv.writer(of, delimiter=' ')
            for word, cde in single_codes:
                w.writerow([cde, word])
        with open('single_word_codes2.txt', 'w') as of:
            w = csv.writer(of, delimiter=' ')
            for word, codes in word2codes.items():
                codes = list(codes)
                if len(codes) == 1:
                    w.writerow([codes[0], word])
        return mr, mr1

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(self.mapping(emb))
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(emb)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred
