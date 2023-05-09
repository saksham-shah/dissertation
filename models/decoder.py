import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, config, output_size, lang=None):
        super(Decoder, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.embedding_size = config["embedding_size"]

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=0 if self.num_layers == 1 else config["dropout"])

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        if lang is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(lang.weights))
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.out = nn.Linear(self.hidden_size, output_size)
        self.attention = Attention(self.hidden_size)

        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, inputs, hidden):
        # Encode input into token embedding
        embedded = self.embedding(inputs.unsqueeze(0))
        embedded = self.embedding_dropout(embedded) # 1, 1, emb

        try:
            embedded = embedded.view(1, inputs.size(0), self.embedding_size)
        except:
            embedded = embedded.view(1, 1, self.embedding_size)

        # Put input through decoder
        output = F.relu(embedded)
        output, hidden = self.lstm(output, hidden) # 1, 1, hidden

        # Compute output prediction
        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden