import itertools
import os

dico_max_sizes = [1000, 500, 5000, 100]
dis_lambdas = [0.1, 1, 0.01]
n_epochses = [40, 80, 20]
epoch_sizes = [30000, 60000, 10000]
tgt_embs = ['../../data/NOTEEVENTS_proc_20_5_128.embed', '../../data/NOTEEVENTS_proc_50_5_128.embed']
dis_stepses = [10, 20, 3, 1]
n_refs = [10, 20, 5]

for opts in itertools.product(dico_max_sizes, dis_lambdas, n_epochses, tgt_embs, epoch_sizes, dis_stepses, n_refs):
    dico_max_size, dis_lambda, n_epochs, tgt_emb, epoch_size, dis_steps, n_ref = opts
    os.system(f"python unsupervised.py --src_lang codes --tgt_lang words --src_emb ../../data/fc_full_all_trained_30_128_10 --tgt_emb {tgt_emb} --emb_dim 128 --epoch_size {epoch_size} --n_refinement {n_ref} --full_vocab --cross_modal --dico_max_size {dico_max_size} --n_epochs {n_epochs} --dis_steps {dis_steps} --dis_lambda {dis_lambda} --cuda true")
