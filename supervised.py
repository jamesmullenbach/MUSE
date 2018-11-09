# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict

from gensim.models import KeyedVectors
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# validation
parser.add_argument("--val_metric", type=str, default="precision_at_1-nn", help="validation metric to choose the best mapping e.g. precision_at_1-csls_knn_10")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Similarity method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S. default=S2T&T2S (mutual nearest neighbors)")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=0, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--subsample", type=float, default=1, help="Fraction of matching pairs to sub-sample to simulate vocabulary mismatch (default 1 = no subsample)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--eval_emb", type=str, default='', help="path to evaluation embeddings for e.g. full-description alignment")
parser.add_argument("--glo_emb", type=str, default="", help="global embeddings file for evaluation")
parser.add_argument("--removed_keys_file", type=str, help="file containing keys removed from site 0. site 1 file will be inferred")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--full_vocab", action="store_true", help="flag to signify the input embeddings comprise the whole vocabulary")
parser.add_argument("--cross_modal", action="store_true", help="flag to signify we're doing cross-modal alignment, so evaluate on the description stuff")
parser.add_argument("--desc_align", action="store_true", help="flag to signify we're doing full-description alignment")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default", "uwc", "owc", "desc"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or (params.dico_max_size < params.dico_max_rank or params.dico_max_rank == 0)
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, eval_emb, mapping, _ = build_model(params, False)
if params.glo_emb:
    glo_emb = KeyedVectors.load_word2vec_format(params.glo_emb) if params.glo_emb else None
else:
    glo_emb = None
trainer = Trainer(src_emb, tgt_emb, glo_emb, eval_emb, mapping, None, params)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train, params.subsample, params.dico_eval)


# define the validation metric
logger.info("Validation metric: %s" % params.val_metric)

"""
Learning loop for Procrustes Iterative Learning
"""
for n_iter in range(params.n_refinement + 1):

    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, params.val_metric, n_iter)
    logger.info('End of iteration %i.\n\n' % n_iter)


# export embeddings
if params.export:
    trainer.reload_best()
    trainer.export(params.cross_modal)

if params.removed_keys_file and params.glo_emb:
    # evaluate global ranking after combining the results in export()
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.global_ranking_eval(to_log)
else:
    to_log = OrderedDict({'n_iter': n_iter})
    mr, mr1 = evaluator.desc_to_code_retrieval_eval(to_log)
    wcmr, wcmr1 = evaluator.word_to_codes_retrieval_eval(to_log)
    with open('sup_results.log', 'a') as af:
        af.write('\t'.join([str(params), str(mr), str(mr1), str(wcmr), str(wcmr1)]) + "\n")
