import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, config, embedding_size):
        super(Encoder, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]

        # self.embedding = nn.Embedding(input_size, embedding_size)
        # if weights_matrix is not None:
        #     self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=0 if self.num_layers == 1 else config["dropout"], bidirectional=config["bidirectional"])

    def forward(self, embedded, input_lengths, restore_indexes, hidden=None):
        # embedded = self.embedding(inputs)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.lstm(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output.index_select(1, restore_indexes)

        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return output, hidden