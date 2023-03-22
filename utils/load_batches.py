import math
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data import *

class Build_Data(Dataset):
    def __init__(self, mwps):
        self.mwps = mwps
        self.len = len(mwps)

    def __getitem__(self, index):
        return {'question': self.mwps[index].question, 'formula': self.mwps[index].equation, 'answer': self.mwps[index].answer, 'numbers': self.mwps[index].numbers}

    def __len__(self):
        return self.len

def batch_data(mwps, batch_size=1):
    dataset = Build_Data(mwps)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def train_test(config, mwps):
    random.seed(1)
    random.shuffle(mwps)

    boundary = math.floor(len(mwps) * 0.9)
    print(len(mwps), boundary)

    train = batch_data(mwps[:boundary], batch_size=config["batch_size"])
    test = batch_data(mwps[boundary:], batch_size=1)
    
    return train, test