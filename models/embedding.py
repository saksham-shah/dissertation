import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, config, input_size, embedding_size, lang=None, weights_matrix=None):
        super(Embedding, self).__init__()

        self.num_emb = config["num_emb"]
        self.lang = lang

        self.embedding = nn.Embedding(input_size, embedding_size)
        if weights_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

    def forward(self, inputs, numbers=None):
        emb = self.embedding(inputs)
        if not self.num_emb or numbers is None:
            return emb

        concat = []

        for input in inputs:
            batches = []
            for batch in range(input.size(0)):
                token = self.lang.index2token[input.item()]
                if token[batch] == '#':
                    index = int(token[1:]) - 1
                    # print(index, token, numbers)
                    batches.append([1, numbers[batch][index]])
                else:
                    batches.append([0, 0])
            concat.append(batches)
        
        concat = torch.tensor(concat, device=device)
        concat = torch.concat([concat, emb], dim=2)

        return concat