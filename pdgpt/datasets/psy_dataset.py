import os.path

from torch.utils.data import Dataset

from pdgpt.utils import read_json


class PsyDataset(Dataset):
    def __init__(self, dataset, cache_dir, split):
        super(PsyDataset, self).__init__()
        self.no_val = True
        self.dataset = read_json(os.path.join(cache_dir, dataset, f'{dataset}.json'))
        self.length = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length
