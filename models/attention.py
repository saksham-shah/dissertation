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
        energy = torch.cat((hidden, encoder_outputs), dim=2) # seq_len, batch, emb+hidden

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