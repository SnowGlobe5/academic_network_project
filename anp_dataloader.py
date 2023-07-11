from datetime import datetime
from typing import Sequence, Union

import pandas as pd
import torch.utils.data

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


class ANPDataLoaderLegacy(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 root="ANP_DATA", fold=-1, max_year=None, keep_edges=False, **kwargs):
        self.dataset = dataset
        self.root = root
        self.fold = fold
        self.max_year = max_year
        self.keep_edges = keep_edges
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, data_list):
        data = self.filter_data(self.dataset[0])
        return data

    def filter_data(self, data):
        subset_dict = {}
        subset_dict_next_year = {}
        if self.fold != -1:
            df_auth = pd.read_csv(f"{self.root}/split/authors_{self.fold}.csv")
            authors_filter_list = df_auth.values.flatten()
            if not self.keep_edges:
                subset_dict['author'] = subset_dict_next_year['author'] = torch.tensor(authors_filter_list)
        else:
            df_auth = pd.read_csv(f"{self.root}/mapping/authors.csv", index_col='id')
            authors_filter_list = df_auth.index
        papers_list_next_year = []
        papers_list_year = []
        for i, row in enumerate(data['paper'].x.tolist()):
            if row[0] <= self.max_year:
                papers_list_year.append(i)
            elif row[0] == self.max_year + 1:
                papers_list_next_year.append(i)
        subset_dict['paper'] = torch.tensor(papers_list_year)
        papers_list_year.extend(papers_list_next_year)
        subset_dict_next_year['paper'] = torch.tensor(papers_list_year)
        return data.subgraph(subset_dict), data.subgraph(subset_dict_next_year), sorted(authors_filter_list.tolist()), papers_list_next_year
