import sys
import rsemcsv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POS_LAB = 1
NEG_LAB = 0
IN_FILE = sys.argv[1]
LABEL_FILE = sys.argv[2]
TRAIN_DAYS = {int(v) for v in sys.argv[3].split(',')}
TEST_DAYS = {int(v) for v in sys.argv[4].split(',')}
K = int(sys.argv[5])

# load the data
label_dict = rsemcsv.read_label_file(LABEL_FILE, 'y')
all_examples,gene_indices = rsemcsv.read_rsem_csv(IN_FILE, label_dict)
ind2gene = {gene_indices[g] : g for g in gene_indices.keys()}

# grab the day specified examples
train_examples = [x for x in all_examples if x.day in TRAIN_DAYS]
test_examples = [x for x in all_examples if x.day in TEST_DAYS]

# sort them into a dictionary for running algos
all_train_comps = {x.name for x in train_examples if x.name.startswith('t') or x.name.startswith('c')} # all T&C compound names
uniq_comps = {x.name for x in test_examples if x.name.startswith('t') or x.name.startswith('c')} # all T&C compound names
train_exdict = rsemcsv.build_compound_example_dict(uniq_comps, train_examples)
test_exdict = rsemcsv.build_compound_example_dict(uniq_comps, test_examples)

# do leave-one-out
pos_prob_dict = dict()
for test_comp in uniq_comps:
    # be sure to exclude this compound from the training set
    test_comps = set([test_comp])
    train_comps = all_train_comps - test_comps
    x_train,y_train = rsemcsv.build_in_out_for_sklearn(train_comps, train_exdict, POS_LAB, NEG_LAB)
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
