import sys
import rsem2dcsv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

POS_LAB = 1
NEG_LAB = 0
IN_FILE = sys.argv[1]
HI_LO = sys.argv[2]
SELK_FILE = sys.argv[3]
TRAIN_DAYS = {int(v) for v in sys.argv[4].split(',')}
TEST_DAYS = {int(v) for v in sys.argv[5].split(',')}

# get the data and the relevant compounds
all_examples,gene_indices = rsem2dcsv.read_rsem_csv(IN_FILE)

# read all the selected genes for folds
fold_keep_inds = dict()
with open(SELK_FILE) as infile:
    for l in infile:
        # first token is held out compound, rest are gene names
        parts = l.strip().split(',')
        name = parts[0]
        inds = [gene_indices[g] for g in parts[1:]]
        fold_keep_inds[name] = sorted(inds)

# filter to the specified days and concentration
train_examples = [e for e in all_examples if e.day in TRAIN_DAYS and e.concentration in {'na', HI_LO}]
test_examples = [e for e in all_examples if e.day in TEST_DAYS and e.concentration in {'na', HI_LO}]
is_toxic = {e.name : e.toxic for e in all_examples}
print(len(train_examples))
print(len(test_examples))
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
    test_ex = [e for e in test_examples if e.name in test_comps]
    # grab the data tensors
    x_train,y_train = rsem2dcsv.build_in_out_for_sklearn(train_ex, POS_LAB, NEG_LAB)
    x_test,y_test = rsem2dcsv.build_in_out_for_sklearn(test_ex, POS_LAB, NEG_LAB)
    # filter to kept indices for this hold out
    keep_inds = fold_keep_inds[test_comp]
    x_train = [[r[ki] for ki in keep_inds] for r in x_train]
    x_test = [[r[ki] for ki in keep_inds] for r in x_test]
    # train a model
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf.fit(x_train, y_train)
    # get the probabilities
    tox_ind = list(clf.classes_).index(POS_LAB) # numpy arrays don't have .index
    # round them to 8 digits
    y_prob = map(lambda p: round(p[tox_ind], 8), clf.predict_proba(x_test)) # tox probs
    prob_avg = np.mean(y_prob)
    pos_prob_dict[test_comp] = prob_avg
    print('{0} -> {1} -> {2}'.format(test_comp, y_prob, prob_avg))

# grab the per-compound labels and predictions
names = sorted(all_test_comps)
y_test = [POS_LAB if is_toxic[c] else NEG_LAB for c in names]
y_prob = [pos_prob_dict[c] for c in names]

# evaluate
fpr,tpr,thresh = roc_curve(y_test, y_prob, drop_intermediate=False)
roc_auc = auc(fpr, tpr)

# and dump results
y_test = map(lambda x: str(x), y_test)
y_prob = map(lambda x: str(x), y_prob)
fpr = map(lambda x: str(x), fpr)
tpr = map(lambda x: str(x), tpr)
print(','.join(['names'] + names))
print(','.join(['labels'] + y_test))
print(','.join(['probs'] + y_prob))
print('roc_auc,{0}'.format(roc_auc))
print(','.join(['fpr'] + fpr))
print(','.join(['tpr'] + tpr))
