import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, weights_matrix=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        if weights_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

    def forward(self, inputs):
        return self.embedding(inputs)