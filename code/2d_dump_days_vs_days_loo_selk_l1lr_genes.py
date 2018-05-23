import sys
import rsem2dcsv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POS_LAB = 1
NEG_LAB = 0
IN_FILE = sys.argv[1]
HI_LO = sys.argv[2]
TRAIN_DAYS = {int(v) for v in sys.argv[3].split(',')}
TEST_DAYS = {int(v) for v in sys.argv[4].split(',')}
K = int(sys.argv[5])

# get the data and the relevant compounds
all_examples,gene_indices = rsem2dcsv.read_rsem_csv(IN_FILE)
ind2gene = {gene_indices[g] : g for g in gene_indices.keys()}

# filter to the specified days and concentration
train_examples = [e for e in all_examples if e.day in TRAIN_DAYS and e.concentration in {'na', HI_LO}]
test_examples = [e for e in all_examples if e.day in TEST_DAYS and e.concentration in {'na', HI_LO}]
is_toxic = {e.name : e.toxic for e in all_examples}
# do leave-one-out
pos_prob_dict = dict()
all_train_comps = {e.name for e in train_examples}
all_test_comps = {e.name for e in test_examples}

# iterate through the test compounds (may differ from train)
for test_comp in sorted(all_test_comps):
    # be sure to exclude this compound from the training set
    test_comps = set([test_comp])
    train_comps = all_train_comps - test_comps
    train_ex = [e for e in train_examples if e.name in train_comps]
    # grab the data tensors
    x_train,y_train = rsem2dcsv.build_in_out_for_sklearn(train_ex, POS_LAB, NEG_LAB)
    # standardize first
    ss = StandardScaler()
    ss.fit(x_train)
    ss_x_train = ss.transform(x_train)
    l1lr = LogisticRegression(C=1.0, penalty='l1')
    l1lr.fit(ss_x_train, y_train)
    # best effort to pick K genes by coefficient magnitude
    th = float('1e-10')
    e10_cf_tups = [(i,c) for i,c in enumerate(l1lr.coef_[0]) if abs(c) > th]
    e10_cf_tups = sorted(e10_cf_tups, key=lambda x: abs(x[1]), reverse=True)
    # print out the selected genes
    k = min(K, len(e10_cf_tups))
    selk_indices = [i for i,c in e10_cf_tups[:k]]
    selk_genes = [ind2gene[i] for i in selk_indices]
    print(','.join([test_comp] + selk_genes))
