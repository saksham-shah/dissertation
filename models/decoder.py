import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, dropout=0.1, weights_matrix=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=0 if num_layers == 1 else dropout)

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        if weights_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs.unsqueeze(0))
        embedded = self.embedding_dropout(embedded) # 1, 1, emb

        try:
            embedded = embedded.view(1, inputs.size(0), self.embedding_size)
        except:
            embedded = embedded.view(1, 1, self.embedding_size)

        embedded, hidden = self.lstm(embedded, hidden) # 1, 1, hidden

        # Calculate the context vector and attention weights
        context, attention_weights = self.attention(embedded, encoder_outputs) # 1, emb

        # Concatenate the context vector and the embedded input
        combined = torch.cat((embedded[0], context), dim=1) # 1, emb+hid
        concat = F.relu(self.concat(combined)) # 1, hidden

        output = self.out(concat)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attention_weights