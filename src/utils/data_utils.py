import random
import numpy as np

import torch
from torch.utils.data import Dataset

from tools import TatoebaDownloader
from utils import LOGGER, colorstr



def get_tatoeba(config):
    downloader = TatoebaDownloader(config)
    trainset, testset = downloader()
    return trainset, testset


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, config, data, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        
        self.text, self.label = [], []
        for text, label in data:
            self.text.append(text)
            self.label.append(label)

        assert len(self.text) == len(self.label)
        self.length = len(self.label)


    def __getitem__(self, idx):
        text, label = self.text[idx], self.label[idx]
        text = self.tokenizer.encode(text)[:self.max_len]
        text += [self.tokenizer.pad_token_id]*(self.max_len-len(text))
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    
    def __len__(self):
        return self.length
    

class CustomDLoader(Dataset):
    def __init__(self, path):
        LOGGER.info(colorstr('red', 'Custom dataloader is required..'))
        raise NotImplementedError

    def __getitem__(self, idx):
        pass
    
    def __len__(self):
        pass