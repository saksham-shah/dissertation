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
        return {'question': self.mwps[index].full_question, 'formula': self.mwps[index].target, 'answer': self.mwps[index].answer}

    def __len__(self):
        return self.len

def batch_data(config, mwps):
    dataset = Build_Data(mwps)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"])
    return train_loader

def train_test(config, mwps, batch_test=False):
    random.seed(1)
    random.shuffle(mwps)

    boundary = math.floor(len(mwps) * 0.9)
    print(len(mwps), boundary)

    train = batch_data(config, mwps[:boundary])
    test = mwps[boundary:]

    if batch_test:
        test = batch_data(config, test)
    
    return train, test