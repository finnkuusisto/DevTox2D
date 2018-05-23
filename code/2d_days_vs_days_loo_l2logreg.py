import sys
import rsem2dcsv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

CS = 10.0**np.arange(-4, 4)
PARAMS = {'C': CS}
POS_LAB = 1
NEG_LAB = 0
IN_FILE = sys.argv[1]
HI_LO = sys.argv[2]
TRAIN_DAYS = {int(v) for v in sys.argv[3].split(',')}
TEST_DAYS = {int(v) for v in sys.argv[4].split(',')}

# get the data and the relevant compounds
all_examples,gene_indices = rsem2dcsv.read_rsem_csv(IN_FILE)

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
    # do normalization by the train set ([0,1] scaling flavor)
    ss = StandardScaler()
    ss.fit(x_train)
    x_train = ss.transform(x_train)
    x_test = ss.transform(x_test)
    # train a model
    lr = LogisticRegression(dual=True)
    clf = GridSearchCV(lr, param_grid=PARAMS, iid=False)
    clf.fit(x_train, y_train)
    # get the probabilities
    tox_ind = list(clf.best_estimator_.classes_).index(POS_LAB) # numpy arrays don't have .index
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
