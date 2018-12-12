# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
import os
from logging import getLogger
from collections import defaultdict
from nltk.tokenize import word_tokenize
import numpy as np
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .dictionary import Dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, glo_emb, eval_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.glo_emb = glo_emb
        self.eval_emb = eval_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False
        self.it = 0
        self.best_it = 0
        self.dico_pairs = []

        self.tok = RegexpTokenizer(r'\w+')
        self.code2desc = {}                                                                                                            
        with open('/data/mimicdata/raw/ICD9_descriptions') as f:                                                                               
            r = csv.reader(f, delimiter='\t')                                  
            for row in r:                                                                                                                      
                codename = 'd_' + row[0].replace('.', '')                      
                self.code2desc[codename] = row[1]

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train, subsample, dico_eval=None):
        """
        Load training dictionary (set of ground truth pairs).
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char" or dico_train == "desc":
            suffix = '_w' if dico_train == "desc" else ""
            self.dico, src_dicts, tgt_dicts = load_identical_char_dico(word2id1, word2id2, subsample, suffix=suffix)
            self.src_dico = Dictionary(*src_dicts, self.params.src_lang)
            self.tgt_dico = Dictionary(*tgt_dicts, self.params.tgt_lang)
            self.params.src_dico = self.src_dico
            self.params.tgt_dico = self.tgt_dico

            if dico_eval:
                #filter out validation pairs
                if self.params.desc_align:
                    #get word-level word2id
                    eval_word2id = self.params.eval_dico.word2id
                _, eval_pairs = load_dictionary(dico_eval, word2id1, eval_word2id, return_pairs=True)
                eval_codes = set([c for c,w in eval_pairs])
                dico_pairs = []
                #create new list of pairs, excluding codes that are in ground truth eval codes
                for i, j in self.dico:
                    src = self.src_dico.id2word[i.item()]
                    tgt = self.tgt_dico.id2word[j.item()]
                    if src not in eval_codes:
                        dico_pairs.append((src, tgt))
                dico = torch.LongTensor(len(dico_pairs), 2)
                for i, (word1, word2) in enumerate(dico_pairs):
                    dico[i, 0] = word2id1[word1]
                    dico[i, 1] = word2id2[word2]
                self.dico = dico

        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # construct dictionary using descriptions
        elif dico_train == "uwc":
            word2codes = defaultdict(set)
            pairs = []
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
                            if tok in word2id2 and cde in word2id1:
                                word2codes[tok].add(cde)
            for word, codes in word2codes.items():
                if len(codes) == 1:
                    pairs.append((list(codes)[0], word))
            pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
            #if dico_eval:
            #    #filter out validation pairs
            #    _, eval_pairs = load_dictionary(dico_eval, word2id1, word2id2, return_pairs=True)
            #    eval_pairs = set(eval_pairs)
            #    pairs = [pair for pair in pairs if pair not in eval_pairs]
            dico = torch.LongTensor(len(pairs), 2)
            for i, (word1, word2) in enumerate(pairs):
                dico[i, 0] = word2id1[word1]
                dico[i, 1] = word2id2[word2]
            self.dico = dico
        # dictionary provided by the user
        else:
            dico, pairs = load_dictionary(dico_train, word2id1, word2id2, return_pairs=True)
            #if dico_eval:
            #    #filter out validation pairs
            #    _, eval_pairs = load_dictionary(dico_eval, word2id1, word2id2, return_pairs=True)
            #    eval_pairs = set(eval_pairs)
            #    pairs = [pair for pair in pairs if pair not in eval_pairs]
            dico = torch.LongTensor(len(pairs), 2)
            for i, (word1, word2) in enumerate(pairs):
                dico[i, 0] = word2id1[word1]
                dico[i, 1] = word2id2[word2]
            self.dico = dico

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

        self.dico_pairs = [(self.src_dico.id2word[i.item()], self.tgt_dico.id2word[j.item()]) for i, j in self.dico]

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings, *adding* to existing dictionary
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        new_dico = build_dictionary(src_emb, tgt_emb, self.params)
        new_dico_pairs = [(self.src_dico.id2word[i.item()], self.tgt_dico.id2word[j.item()]) for i, j in new_dico]
        for pair in new_dico_pairs:
            if pair not in self.dico_pairs:
                code, word = pair
                if code in self.code2desc:
                    toks = [t.lower() for t in self.tok.tokenize(self.code2desc[code])]
                    if word in toks:
                        self.dico_pairs.append(pair)
                self.dico_pairs.append(pair)
        self.dico_pairs = sorted(self.dico_pairs, key=lambda x: self.src_dico.word2id[x[0]])
        self.dico = torch.LongTensor(len(self.dico_pairs), 2)
        for i, (word1, word2) in enumerate(self.dico_pairs):
            self.dico[i, 0] = self.src_dico.word2id[word1]
            self.dico[i, 1] = self.tgt_dico.word2id[word2]
        logger.info('New train dictionary of %i pairs.' % self.dico.size(0))

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric, n_iter):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        self.it = n_iter
        if to_log[metric] > self.best_valid_metric:
            self.best_it = n_iter
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from iteration %d at %s ...' % (self.best_it, path))
        print('* Reloading the best model from iteration %d at %s ...' % (self.best_it, path))
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self, cross_modal):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, mode='src', full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, mode='tgt', full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        #combine vocabularies
        common = set(self.src_dico.word2id.keys()).intersection(set(self.tgt_dico.word2id.keys()))
        self.src_dico.word2id = {(word[:-3] if word.endswith('_s1') else word):ix for word, ix in self.src_dico.word2id.items()}
        self.tgt_dico.word2id = {(word[:-3] if word.endswith('_s2') else word):ix for word, ix in self.tgt_dico.word2id.items()}
        full_vocab = sorted(set(self.src_dico.word2id.keys()) \
                    .union(set(self.tgt_dico.word2id.keys())))

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        #put mapping on cpu for simplicity, so we don't put every loaded emb on gpu individually
        self.mapping.cpu()
        if not cross_modal:
            E1 = Variable(torch.zeros((len(full_vocab), src_emb.shape[1])))
            E2 = Variable(torch.zeros((len(full_vocab), src_emb.shape[1])))
            id2word, word2id = {}, {}
            for i, word in tqdm(enumerate(full_vocab)):
                if word in self.src_dico.word2id.keys():
                    E1[i] = self.mapping(Variable(src_emb[self.src_dico.word2id[word]]))
                else:
                    E1[i] = tgt_emb[self.tgt_dico.word2id[word]]
                if word in self.tgt_dico.word2id.keys():
                    E2[i] = tgt_emb[self.tgt_dico.word2id[word]]
                else:
                    E2[i] = self.mapping(Variable(src_emb[self.src_dico.word2id[word]]))
                id2word[i] = word
                word2id[word] = i
            src_dico = Dictionary(id2word, word2id, self.params.src_lang)
            tgt_dico = Dictionary(id2word, word2id, self.params.tgt_lang)
        else:
            E1 = Variable(torch.zeros((len(self.src_dico.word2id)), src_emb.shape[1]))
            E2 = Variable(torch.zeros((len(self.tgt_dico.word2id)), src_emb.shape[1]))
            id2src, src2id = {}, {}
            id2tgt, tgt2id = {}, {}
            for i, (word, ix) in tqdm(enumerate(sorted(self.src_dico.word2id.items(), key=lambda x: x[0]))):
                E1[i] = self.mapping(Variable(src_emb[self.src_dico.word2id[word]]))
                id2src[i] = word
                src2id[word] = i
            for i, (word, ix) in tqdm(enumerate(sorted(self.tgt_dico.word2id.items(), key=lambda x: x[0]))):
                E2[i] = tgt_emb[self.tgt_dico.word2id[word]]
                id2tgt[i] = word
                tgt2id[word] = i
            src_dico = Dictionary(id2src, src2id, self.params.src_lang)
            tgt_dico = Dictionary(id2tgt, tgt2id, self.params.tgt_lang)
        src_emb = E1.clone()
        tgt_emb = E2.clone()

        #reassign dicts
        params.src_dico = src_dico
        params.tgt_dico = tgt_dico
        self.src_dico = src_dico
        self.tgt_dico = tgt_dico
        self.params.src_dico = src_dico
        self.params.tgt_dico = tgt_dico
        self.src_emb = torch.nn.Embedding.from_pretrained(src_emb)
        self.tgt_emb = torch.nn.Embedding.from_pretrained(tgt_emb)
        if params.cuda:
            self.src_emb.cuda()
            self.tgt_emb.cuda()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
