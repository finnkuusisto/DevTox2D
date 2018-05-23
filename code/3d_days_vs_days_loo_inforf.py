import sys,rsemcsv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def inforf_predict_for_one(train_comps, test_comp, train_exdict, test_exdict, pos_lab, neg_lab):
    x_train,y_train = rsemcsv.build_in_out_for_sklearn(train_comps, train_exdict, POS_LAB, NEG_LAB)
    x_test,y_test = rsemcsv.build_in_out_for_sklearn([test_comp], test_exdict, POS_LAB, NEG_LAB)
    # train a model
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf.fit(x_train, y_train)
    # get the probabilities for test
    tox_ind = list(clf.classes_).index(pos_lab) # numpy arrays don't have .index
    # round them to 8 digits
    y_preds = map(lambda p: round(p[tox_ind], 8), clf.predict_proba(x_test)) # tox probs
    return y_preds

POS_LAB = 1
NEG_LAB = 0
IN_FILE = sys.argv[1]
LABEL_FILE = sys.argv[2]
TRAIN_DAYS = {int(v) for v in sys.argv[3].split(',')}
TEST_DAYS = {int(v) for v in sys.argv[4].split(',')}
DVD = sys.argv[3] + 'v' + sys.argv[4]

# load the data
label_dict = rsemcsv.read_label_file(LABEL_FILE, 'y')
all_examples,gene_indices = rsemcsv.read_rsem_csv(IN_FILE, label_dict)

# grab the day specified examples
train_examples = [x for x in all_examples if x.day in TRAIN_DAYS]
test_examples = [x for x in all_examples if x.day in TEST_DAYS]

# sort them into a dictionary for running algos
uniq_comps = {x.name for x in all_examples if x.name.startswith('t') or x.name.startswith('c')} # all T&C compound names
train_exdict = rsemcsv.build_compound_example_dict(uniq_comps, train_examples)
test_exdict = rsemcsv.build_compound_example_dict(uniq_comps, test_examples)

# do leave-one-out
pos_prob_dict = dict()
for test_comp in uniq_comps:
    train_comps = uniq_comps - set([test_comp])
    # get the prob for the specified day
    day_preds = inforf_predict_for_one(train_comps, test_comp, train_exdict, test_exdict, POS_LAB, NEG_LAB)
    pos_prob_dict[test_comp] = np.mean(day_preds)
    print('{0}={{"d{1}":{2},"final":{3}}}'.format(test_comp, DVD, day_preds, np.mean(day_preds)))

# now grab the per-compound labels and predictions
names = sorted(uniq_comps)
y_test = map(lambda c: POS_LAB if label_dict[c] else NEG_LAB, names)
y_prob = map(lambda c: pos_prob_dict[c], names)

# evaluate
fpr,tpr,thresh = roc_curve(y_test, y_prob)
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
