import math
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.load_data import *

# torch Dataset class for batching
class Build_Data(Dataset):
    def __init__(self, mwps, rpn=False):
        self.mwps = mwps
        self.len = len(mwps)
        self.rpn = rpn

    def __getitem__(self, index):
        equation = self.mwps[index].equation
        if self.rpn:
            equation = " ".join(infix_to_rpn(equation.split(" ")))
        return {
            'question': self.mwps[index].question,
            'formula': equation,
            'answer': self.mwps[index].answer,
            'numbers': self.mwps[index].numbers
        }

    def __len__(self):
        return self.len

def batch_data(mwps, rpn=False, batch_size=1):
    dataset = Build_Data(mwps, rpn)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# Split into folds for cross-validation
def train_test(mwps, fold=0, nfold=10):
    left_bound = math.floor(len(mwps) * fold / nfold)
    right_bound = math.floor(len(mwps) * (fold + 1) / nfold)

    left = mwps[:left_bound]
    fold = mwps[left_bound:right_bound]
    right = mwps[right_bound:]

    return left + right, fold