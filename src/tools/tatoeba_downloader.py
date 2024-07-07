import os
import random
from tqdm import tqdm

from utils import LOGGER
from utils.func_utils import preprocessing
from utils.filesys_utils import read_dataset, write_dataset



class TatoebaDownloader:
    def __init__(self, config):
        self.data_dir = config.tatoeba.path
        self.splits = ['train', 'test']

    @staticmethod
    def preprocessing_data(r_path):
        with open(r_path, 'r') as f:
            data = f.readlines()
    
        data = [d.split('\t')[:2] for d in data]
        data = [[preprocessing(d[0]), preprocessing(d[1])] for d in tqdm(data)]

        random.seed(999)
        all_id = list(range(len(data)))
        testset_id = random.sample(all_id, 1000)
        trainset_id = list(set(all_id) - set(testset_id))
        trainset = [data[i] for i in trainset_id]
        testset = [data[i] for i in testset_id]

        return trainset, testset
        

    @staticmethod
    def is_exist(path):
        return os.path.isfile(path)

    
    def __call__(self):
        raw_data_path = os.path.join(self.data_dir, 'tatoeba/raw/eng-fra.txt')
        pp_trainset_path, pp_testset_path = os.path.join(self.data_dir, 'tatoeba/processed/eng-fra.train'), os.path.join(self.data_dir, 'tatoeba/processed/eng-fra.test')

        if not (self.is_exist(pp_trainset_path) and self.is_exist(pp_testset_path)):
            LOGGER.info('Pre-processing the raw tatoeba dataset..')
            os.makedirs(os.path.dirname(pp_trainset_path), exist_ok=True)
            trainset, testset = self.preprocessing_data(raw_data_path)
            write_dataset(pp_trainset_path, trainset)
            write_dataset(pp_testset_path, testset)            

            return trainset, testset
        
        return read_dataset(pp_trainset_path), read_dataset(pp_testset_path)