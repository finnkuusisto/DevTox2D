import sys
import rsem2dcsv

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

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
    # select k best by rfe
    mms = MinMaxScaler()
    mms.fit(x_train)
    mm_x_train = mms.transform(x_train)
    rfe_est = SVC(C=1.0, kernel='linear')
    rfe = RFE(rfe_est, n_features_to_select=K, step=0.01)
    rfe.fit(mm_x_train, y_train)
    # print out the selected genes
    selk_indices = rfe.get_support(indices=True)
    selk_genes = [ind2gene[i] for i in selk_indices]
    print(','.join([test_comp] + selk_genes))
