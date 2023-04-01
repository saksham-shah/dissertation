import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden, encoder_outputs): # seq_len, batch, emb
        batch_size = encoder_outputs.shape[1]
        sequence_length = encoder_outputs.shape[0]
        # Create a new tensor with the same size as encoder outputs
        hidden = hidden.repeat(sequence_length, 1, 1) # seq_len, batch, hidden
        # hidden = hidden.expand(sequence_length, -1, -1)

        # Concatenate the hidden state with each encoder output
        energy = torch.cat((hidden, encoder_outputs), dim=2) # seq_len, batch, 2*hidden

        # Calculate the attention weights (energies)
        energy = self.attention(energy).tanh() # seq_len, batch, hidden
        energy = energy.permute(1, 2, 0) # batch, hidden, seq_len
        v = self.v.repeat(batch_size, 1).unsqueeze(1) # batch, 1, hidden
        attention_weights = torch.bmm(v, energy).squeeze(1) # 1, seq_len

        # Normalize the attention weights
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Calculate the weighted sum of the encoder outputs # 1, 1, seq_len and 1, seq_len, emb
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs.permute(1,0,2))
        context = context.squeeze(1) # 1, emb

        return context, attention_weights

class AttnDecoder(nn.Module):
    def __init__(self, config, output_size, lang=None):
        super(AttnDecoder, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.embedding_size = config["embedding_size"]

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=0 if self.num_layers == 1 else config["dropout"])

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        if lang is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(lang.weights))
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.out = nn.Linear(self.hidden_size, output_size)
        self.attention = Attention(self.hidden_size)

        self.concat = nn.Linear(self.hidden_size + self.embedding_size, self.hidden_size)


    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs.unsqueeze(0))
        embedded = self.embedding_dropout(embedded) # 1, 1, emb

        try:
            embedded = embedded.view(1, inputs.size(0), self.embedding_size)
        except:
            embedded = embedded.view(1, 1, self.embedding_size)

        # embedded: 1, N, emb

        # Calculate the context vector and attention weights
        context, attention_weights = self.attention(hidden[0], encoder_outputs) # 1, emb

        # Concatenate the context vector and the embedded input
        combined = torch.cat((embedded[0], context), dim=1) # 1, emb+hid
        concat = F.relu(self.concat(combined)).unsqueeze(0) # 1, hidden

        output, hidden = self.lstm(concat, hidden)

        output = self.out(output.squeeze(0))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attention_weights