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
    def __init__(self, config, data, tokenizers):
        self.data = data
        self.src_tokenizer, self.trg_tokenizer = tokenizers[0], tokenizers[1]
        self.max_len = config.max_len
        
        self.all_src, self.all_trg, self.all_mask = [], [], []
        for src, trg in self.data:
            s, s_l = self.add_special_token(src, self.src_tokenizer)
            t, _ = self.add_special_token(trg, self.trg_tokenizer)
            self.all_src.append(s)
            self.all_trg.append(t)
            mask = torch.zeros(self.max_len)
            mask[:s_l] = 1
            self.all_mask.append(mask)

        assert len(self.all_src) == len(self.all_trg)
        self.length = len(self.all_src)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.bos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        mask_l = len(s)
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s, mask_l


    def __getitem__(self, idx):
        return torch.LongTensor(self.all_src[idx]), torch.LongTensor(self.all_trg[idx]), torch.FloatTensor(self.all_mask[idx])

    
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