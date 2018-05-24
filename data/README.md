# DevTox2D Data

## Expression Data
These are the gene expression profile datasets collected from 3D and 2D human neural tissue models of exposure to various compounds (see Compound Information). The datasets are in transcripts per million (TPM) and each row represents a single sample while columns represent genes. Each sample represents tissue exposure to a single toxic or non-toxic compound at a single exposure length. Sample names are in the first column and identify the compound and exposure length, with differing naming formats for the 2D and 3D datasets. The naming formats are described below.

### expression_2d.csv
In the 2D tissue model dataset, sample names follow two different patterns for the toxic and non-toxic compound exposures:

- `D<num_days_exposure>-C<compound_id_number>` for the non-toxic compounds
- `D<num_days_exposure>Hi-<compound_id_number>` for the toxic compounds

For example, the sample name `D39-C15` corresponds with the non-toxic compound `c15` at 39 days of exposure. The sample name `D27Hi-4` corresponds with the toxic compound `t4` at 27 days of exposure.

### expression_3d.csv
In the 3D tissue model dataset, sample names follow a common pattern. In contrast with the 2D dataset, the 3D dataset also contains two biological replicates of each sample. The naming scheme is as follows:

- `d<num_days_exposure><compound_id><replicate_index>`

For example, the sample name `d2c15b` corresponds with non-toxic compound `c15` at two days of exposure, biological replicate two. The sample name `d7t4a` corresponds with toxic compound `t4` at seven days of exposure, biological replicate one.

## Compound Information
### compound_labels.csv
This file contains the ground truth labels (toxic vs non-toxic) for each of the compound IDs. Note that in this case, the compound IDs as determine the ground truth label as the IDs either start with the letter 't' for toxic or 'c' for non-toxic.

### compound_names.csv
This file contains a map between compound IDs and the actual compound names.

## Feature Selection Directories
These directories contain the precomputed per-fold feature selection results from the paper. The filenames contain information regarding the feature selection algorithm used and number of features selected. Filenames contain the tokens `l1lr`, `minfo`, and `rfe`, which correspond to L1 regularized logistic regression (lasso), mutual information, and recursive feature elimination respectively. Filenames also contain a token that specifies the number of features selected, such as `k300` to indicate 300 features selected.

Each file contains genes selected on a per-fold basis according to leave-one-compound-out cross-validation. Each line in the file follows this format:
```
<held_out_compound>,<selected_gene>,<selected_gene>,<selected_gene>...
```
Where `<held_out_compound>` indicates which compound was not included in the model for feature selection. These selected genes are thus used when evaluating on the held out compound. For example:
```
t4,SEZ6,MARCH2,WWC1,DDN,CDH5,SLITRK6,ANKRD23,HIST1H3H,PSMG4,SNUPN
```
This line indicates that the subsequent genes in the list were selected to be used when `t4` samples were held out of the model. That is, these genes will be used when building a model that is trained on all other compounds but evaluated on `t4`.