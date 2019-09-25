"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
"""
import os
import sys
import click
import random
import collections

# There are 13 integer features and 26 categorical features
continous_features = range(0, 13)
categorial_features = range(13, 39)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                data_fields = line.rstrip('\n').split('\t')
                features = data_fields[1:]
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min_val = [99999] * num_feature
        self.max_val = [-99999] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                data_fields = line.rstrip('\n').split('\t')
                features = data_fields[1:]
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        if self.min_val[i] > val:
                            self.min_val[i] = val
                        if self.max_val[i] < val:
                            self.max_val[i] = val

    def gen(self, idx, val):
        if val == '':
            return 0.0
        eps = 1e-6
        min_v = self.min_val[idx]
        max_v = self.max_val[idx]
        val = (float(val) - min_v) / (max_v - min_v + eps)
        return val


# @click.command("preprocess")
# @click.option("--datadir", type=str, help="Path to raw criteo dataset")
# @click.option("--outdir", type=str, help="Path to save the processed data")
def preprocess(datadir, outdir, train_file='train.txt', test_file='test_file', feature_sizes_file = 'feature_sizes.txt', cutoff=200):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """

    print("building dictionary....")
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, train_file), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, train_file), categorial_features, cutoff=cutoff)

    dict_sizes = dicts.dicts_sizes()

    with open(os.path.join(outdir, feature_sizes_file), 'w') as feature_sizes:
        sizes = [1] * len(continous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training.
    print("transforming {}".format(train_file))
    with open(os.path.join(outdir, train_file), 'w') as out_train:
        with open(os.path.join(datadir, train_file), 'r') as f:
            for line in f:
                data_fields = line.rstrip('\n').split('\t')
                label = data_fields[0]
                features = data_fields[1:]

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i]])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')
                    

    print("transforming {}".format(test_file))
    with open(os.path.join(outdir, test_file), 'w') as out:
        with open(os.path.join(datadir, test_file), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i]])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')

if __name__ == "__main__":
    #preprocess('./data/raw', './data', train_file='train_large.txt', test_file='test_large.txt', feature_sizes_file="feature_sizes_large.txt")
    preprocess('./data/raw', './data', train_file='train.txt', test_file='test.txt', feature_sizes_file = 'feature_sizes.txt',  cutoff=0)
