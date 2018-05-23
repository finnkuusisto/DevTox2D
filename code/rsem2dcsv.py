import sys, re

class DevToxExample:
    def __init__(self, fullname, name, day, toxic, concentration, data):
        self.fullname = fullname
        self.name = name
        self.day = day
        self.toxic = toxic
        self.concentration = concentration
        self.data = data
    def to_str(self):
        return 'full={0}\tname={1}\tday={2}\ttox={3}\tconc={4}'.format(self.fullname, self.name, self.day, self.toxic, self.concentration)

def read_rsem_csv(filename):
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
            fullname = line[0].lower()
            # looks something like "controld0", "d1-c1", "d1lo-1", or "d1hi-1"
            # ones with hi or lo are toxic
            comp_name = ''
            day = 0
            toxic = False
            conc = 'na'
            if not fullname.startswith('d'):
                d_ind = fullname.index('d')
                comp_name = fullname[:d_ind]
                day = int(fullname[d_ind+1:])
            else:
                parts = fullname.split('-')
                # inconsistent naming scheme between tox and control
                if 'hi' in parts[0] or 'lo' in parts[0]: # toxic
                    conc = (parts[0])[-2:] # looks like "D1Lo"
                    toxic = True
                    day = int((parts[0])[1:-2])
                    comp_name = 't' + parts[1] # to be consistent with 3D
                else: # control
                    day = int((parts[0])[1:])
                    comp_name = parts[1]
            # and convert the expression data to floats
            data = map(lambda v: float(v), line[1:])
            # now store our example
            e = DevToxExample(fullname, comp_name, day, toxic, conc, data)
            all_examples.append(e)
    return all_examples,gene_indices

# build the input matrix and output vector for sklearn from comp names
def build_in_out_for_sklearn(examples, pos_lab, neg_lab):
    # just put together a 2d and 1d list
    x = list()
    y = list()
    for e in examples:
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
