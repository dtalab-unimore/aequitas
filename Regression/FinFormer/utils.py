
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
import scipy.sparse as sp
from torch_geometric.utils import to_undirected


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)



def edgeIndexTransform(adj):
    tmp_coo = sp.coo_matrix(adj)
    indices = np.vstack((tmp_coo.row, tmp_coo.col))
    edge_index = torch.LongTensor(indices)
    edge_index = to_undirected(edge_index)
    return edge_index


def pprint(*args):
    print(*args, flush=True)
