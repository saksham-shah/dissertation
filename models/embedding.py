import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, config, input_size, lang=None):
        super(Embedding, self).__init__()

        self.num_emb = config["num_emb"]
        self.lang = lang

        self.embedding = nn.Embedding(input_size, config["embedding_size"])
        if lang is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(lang.weights))

    def forward(self, inputs, numbers=None, restore_indexes=None):
        emb = self.embedding(inputs)
        if not self.num_emb or numbers is None:
            return emb

        concat = []

        # Encode numerical value
        for input in inputs:
            batches = []
            for batch in range(input.size(0)):
                token = self.lang.index2token[input[batch].item()]
                if token[0] == '#':
                    index = int(token[1:])
                    batch = restore_indexes[batch] # get numeric value
                    try:
                        batches.append([1, numbers[batch][index]])
                    except IndexError:
                        batches.append([0, 0])
                else: # if not number, just use a default value of 0
                    batches.append([0, 0])
            concat.append(batches)
        
        concat = torch.tensor(concat, device=device, dtype=torch.float32)
        concat = torch.concat([concat, emb], dim=2)

        return concat