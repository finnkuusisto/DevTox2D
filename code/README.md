# DevTox2D Code
These scripts run four different common off-the-shelf machine learning algorithms (random forest, logistic regression, linear SVM, and multinomial Naive Bayes) on the two different expression datasets included in this repository. Additionally, the scripts run three different common off-the-shelf feature selection algorithms to be used in combination with the aforementioned prediction algorithms. All use leave-one-compound-out cross validation.

*Please excuse the extensive code duplication.*

## 2D Algorithm Scripts
### Basic Algorithms
There are four main scripts that run the basic algorithms on the expression data from the 2D tissue models:

- `2d_days_vs_days_loo_inforf.py` for random forest using an infogain splitting criterion
- `2d_days_vs_days_loo_l2logreg.py` for logistic regression using L2 regularization
- `2d_days_vs_days_loo_linsvm.py` for linear SVM
- `2d_days_vs_days_loo_multinb.py` for multinomial Naive Bayes

All of these scripts have the following command line call signature:
```
python 2d_days_vs_days_loo_x.py <expression_file> <hi_lo> <train_days_list> <test_days_list>
```

Where the arguments mean:

- `<expression_file>` this is the path to the expression data file (see data directory)
- `<hi_lo>` this will always be `hi` for the purposes of this work
- `<train_days_list>` this is a comma-separated list of compound exposure days to include in the training set
- `<test_days_list>` this is a comma-separate list of compound exposure days to include in the testing set

For example, the following is an acceptable call:
```
python 2d_days_vs_days_loo_linsvm.py ../data/expression_2d.csv hi 1,2 3,4
```
This runs a linear SVM on the provided expression data, training the model (using leave-one-compound_out cross-validation) on the data from compound exposure lengths of 1 and 2 days, and then testing the model on the data from compound exposure lengths of 3 and 4 days.

### Basic Algorithms with Feature Selection
These are the same four algorithm scripts mentioned above with the addition of one argument to specify selected features for each fold (see data directory README). That is, these scripts don't do the feature selection themselves but instead accept precomputed selected features. The corresponding script names for each algorithm are the same with `_spec_selk.py` at the end.

All of the scripts have the following command line call signature:
```
python 2d_days_vs_days_loo_x.py <expression_file> <hi_lo> <feature_selection_file> <train_days_list> <test_days_list>
```

Where the new argument as compared to previously mentioned means:

- `<feature_selection_file>` this is the path to the file containing selected features per fold (see data directory)

### Feature Selection Algorithms
There are three scripts that precompute selected features for each leave-one-compound-out cross-validation fold for the 2D tissue models:

- `2d_dump_days_vs_days_loo_selk_l1lr_genes.py` for feature selection by L1 regularized logistic regression (Lasso)
- `2d_dump_days_vs_days_loo_selk_minfo_genes.py` for feature selection by mutual information
- `2d_dump_days_vs_days_loo_selk_rfe_genes.py` for feature selection by recursive feature elimination

All of these scripts have the following command line call signature:
```
python 2d_days_vs_days_loo_x.py <expression_file> <hi_lo> <train_days_list> <test_days_list> <num_features>
```

Where the new argument as compared to the basic algorithms is:

- `<num_features>` this is the number of features to select for each cross-validation fold

This writes output to standard out in the format of the per-fold feature selection files described in the data directory.

## 3D Algorithm Scripts
TODO

## Interpreting Results
The output of the basic algorithm scripts (2D and 3D, with and without feature selection) goes to standard out. Most of the initial output is related to predictions it makes at each fold of the cross-validation and can be ignored, but the last six lines are what users will be most interested in. The following is an example of the last six lines:
```
names,c1,c10,c11,c12,c13,c14,c15,c16,c2,c3,c4,c5,c6,c7,c8,c9,t1,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t2,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t3,t4,t5,t6,t7,t8,t9
labels,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
probs,0.23845936,0.26858503499999997,0.251904925,0.25334599999999996,0.278041985,0.26454119,0.29203269,0.13654180500000002,0.27924107,0.266782295,0.27807214,0.274647705,0.25870083499999996,0.27286039,0.22477296000000002,0.208364145,0.34777154,0.35932388000000004,0.940855075,0.40583777,0.981909725,0.69449302,0.40744813,0.763810085,0.635872475,0.39468391999999997,0.42160222,0.382942375,0.54726562,0.7180346049999999,0.673649055,0.413621125,0.38959647,0.300465065,0.16024388,0.69644762,0.39185128999999996,0.40880605999999997,0.38936804,0.5297229450000001,0.2572208,0.358483785,0.37664295999999997,0.37922653,0.36822489
roc_auc,0.946120689655
fpr,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.625,0.625,0.6875,0.75,0.8125,0.875,0.9375,0.9375,1.0
tpr,0.034482758620689655,0.06896551724137931,0.10344827586206896,0.13793103448275862,0.1724137931034483,0.20689655172413793,0.2413793103448276,0.27586206896551724,0.3103448275862069,0.3448275862068966,0.3793103448275862,0.41379310344827586,0.4482758620689655,0.4827586206896552,0.5172413793103449,0.5517241379310345,0.5862068965517241,0.6206896551724138,0.6551724137931034,0.6896551724137931,0.7241379310344828,0.7586206896551724,0.7931034482758621,0.8275862068965517,0.8620689655172413,0.896551724137931,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9310344827586207,0.9655172413793104,0.9655172413793104,0.9655172413793104,0.9655172413793104,0.9655172413793104,0.9655172413793104,1.0,1.0
```
The first line, preceded with the `names` token, is an ordered list of the compounds evaluated on this run. The line preceded with `labels` is an ordered list of the ground truth labels assigned to those compounds evaluated (`1` is toxic and `0` is non-toxic - see data directory). The `probs` line is an ordered list of the predictions assigned to each of the evaluated compounds. The line preceded with `roc_auc` is the area under the receiver operating characteristic (ROC) curve for this run. The next two lines, preceded by `fpr` and `tpr`, are the points for plotting the ROC curve.