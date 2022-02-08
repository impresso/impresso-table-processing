import sys

sys.path.append("../helpers")
from impresso_id import *


def get_weights_across_dimension(df, column):
    count = df.groupby(column).count()['id']
    weights = [df.shape[0] / (len(count) * count[x]) for x in df[column]]  # w_j=n_samples/(n_classes*n_samples_j)
    return weights