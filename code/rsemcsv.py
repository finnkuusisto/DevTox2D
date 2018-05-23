import sys, re

class DevToxExample:
    def __init__(self, fullname, name, day, toxic, replicate, data):
        self.fullname = fullname
        self.name = name
        self.day = day
        self.toxic = toxic
        self.replicate = replicate
        self.data = data
    def to_str(self):
        return 'full={0}\tname={1}\tday={2}\ttox={3}\trep={4}'.format(self.fullname, self.name, self.day, self.toxic, self.replicate)

def read_label_file(filename, pos_token):
    ret = dict()
    with open(filename) as incsv:
        lines = incsv.readlines()
        header = lines[0].strip().split(',')
        drg_ind = header.index('drug')
        lab_ind = header.index('toxic')
        for line in lines[1:]:
            line = line.strip().split(',')
            ret[line[drg_ind]] = line[lab_ind] == pos_token
    return ret

def read_rsem_csv(filename, label_dict):
    all_examples = list()
    gene_indices = dict()
    with open(filename) as incsv:
        lines = incsv.readlines()
        # read in the header and pull the gene indices
        header = lines[0].strip().replace('"', '').split(',')
        # create the gene index map
        for i,g in enumerate(header[1:]):
            gene_indices[g] = i
        # the rest are the samples
        for line in lines[1:]:
            line = line.strip().replace('"', '').split(',')
            # fullname is just the whole coded name
            fullname = line[0]
            # the toxic/control key starts at a t, c, b, or u
            tox_start = re.search('[tcbu]', fullname).start()
            # and ends at an a or b
            rep_start = tox_start + 1 + re.search('[abcd]', fullname[tox_start+1:]).start()
            # grab the day component (always starts with d)
            day = int(fullname[1:tox_start]) # skip d char
            # grab the name and tox component
            comp_name = fullname[tox_start:rep_start]
            toxic = label_dict[comp_name]
            # and grab the replicate
            replicate = fullname[rep_start:]
            # and convert the expression data to floats
            data = map(lambda v: float(v), line[1:])
            # now store our example
            e = DevToxExample(fullname, comp_name, day, toxic, replicate, data)
            all_examples.append(e)
    return all_examples,gene_indices

def build_compound_example_dict(compound_names, examples):
        ex_dict = dict()
        for comp_name in compound_names:
            comp_ex = [x for x in examples if x.name == comp_name]
            ex_dict[comp_name] = comp_ex
        return ex_dict

def aggregate_samples_by_gene(examples, gene_indices, gene_name):
    gene_index = gene_indices[gene_name]
    samples = [x.data[gene_index] for x in examples]
    return samples

# build the input matrix and output vector for sklearn from comp names
def build_in_out_for_sklearn(compound_names, example_dict, pos_lab, neg_lab):
    # just put together a 2d and 1d list
    x = list()
    y = list()
    for n in compound_names:
        n_exs = example_dict[n]
        for e in n_exs:
            x.append(e.data)
            y.append(pos_lab if e.toxic else neg_lab)
    return x,y

# reduce the sklearn features to those specified
def reduce_sklearn_x_to_chosen_features(x, include_indices):
    # sort just in case
    include_indices = sorted(include_indices)
    newx = list()
    for r in x:
        newr = map(lambda i: r[i], include_indices)
        newx.append(newr)
    return newx
