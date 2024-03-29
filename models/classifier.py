import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.attention import Attention
from models.embedding import Embedding

from utils.prepare_tensors import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, config, q_input_size, a_input_size, q_lang=None, a_lang=None, hidden_1=256, hidden_2=128):
        super(Classifier, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.bidirectional = config["bidirectional"]

        self.q_embedding = Embedding(config, q_input_size, q_lang)
        self.a_embedding = Embedding(config, a_input_size, a_lang)

        self.q_encoder = Encoder(config)
        self.a_encoder = Encoder(config)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, q_input, a_input, q_lengths, a_lengths, numbers, hidden=None):
        q_sorted_input, q_sorted_lengths, q_restore_indexes = sort_by_length(q_input, q_lengths)
        a_sorted_input, a_sorted_lengths, a_restore_indexes = sort_by_length(a_input, a_lengths)

        q_sorted_input = self.q_embedding(q_sorted_input, numbers, q_restore_indexes)
        a_sorted_input = self.a_embedding(a_sorted_input, numbers, a_restore_indexes)

        q_output, q_hidden = self.q_encoder(q_sorted_input, q_sorted_lengths, q_restore_indexes)
        a_output, a_hidden = self.a_encoder(a_sorted_input, a_sorted_lengths, a_restore_indexes)

        q_decoder_hidden = (q_hidden[0][:self.q_encoder.num_layers], q_hidden[1][:self.q_encoder.num_layers])
        a_decoder_hidden = (a_hidden[0][:self.a_encoder.num_layers], a_hidden[1][:self.a_encoder.num_layers])

        context = torch.concat([q_decoder_hidden[0], a_decoder_hidden[0]], dim=2).squeeze(0)

        output = torch.relu(self.fc1(context))
        output = torch.relu(self.fc2(output))
        output = torch.sigmoid(self.out(output)).squeeze(1)
        return output

class AttnClassifier(nn.Module):
    def __init__(self, config, q_input_size, a_input_size, q_lang=None, a_lang=None, hidden_1=256, hidden_2=128):
        super(AttnClassifier, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.bidirectional = config["bidirectional"]

        self.q_embedding = Embedding(config, q_input_size, q_lang)
        self.a_embedding = Embedding(config, a_input_size, a_lang)

        self.q_encoder = Encoder(config)
        self.a_encoder = Encoder(config)

        self.attention = Attention(self.hidden_size)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, q_input, a_input, q_lengths, a_lengths, numbers, hidden=None):
        # Put inputs through encoder
        q_sorted_input, q_sorted_lengths, q_restore_indexes = sort_by_length(q_input, q_lengths)
        a_sorted_input, a_sorted_lengths, a_restore_indexes = sort_by_length(a_input, a_lengths)

        q_sorted_input = self.q_embedding(q_sorted_input, numbers, q_restore_indexes)
        a_sorted_input = self.a_embedding(a_sorted_input, numbers, a_restore_indexes)

        q_output, q_hidden = self.q_encoder(q_sorted_input, q_sorted_lengths, q_restore_indexes)
        a_output, a_hidden = self.a_encoder(a_sorted_input, a_sorted_lengths, a_restore_indexes)

        # Initialise decoder and compute attention weights
        q_decoder_hidden = (q_hidden[0][:self.q_encoder.num_layers], q_hidden[1][:self.q_encoder.num_layers])
        a_decoder_hidden = (a_hidden[0][:self.a_encoder.num_layers], a_hidden[1][:self.a_encoder.num_layers])

        q_attention_weights = self.attention(q_decoder_hidden[0], q_output)
        a_attention_weights = self.attention(a_decoder_hidden[0], a_output)

        # Construct context vector
        q_context = torch.bmm(q_attention_weights.unsqueeze(1), q_output.permute(1, 0, 2))
        a_context = torch.bmm(a_attention_weights.unsqueeze(1), a_output.permute(1, 0, 2))
        q_context = q_context.permute(1, 0, 2) # 1, batch, hidden
        a_context = a_context.permute(1, 0, 2) # 1, batch, hidden

        context = torch.concat([q_context, a_context], dim=2).squeeze(0)

        # Put through fully-connected layers
        output = torch.relu(self.fc1(context))
        output = torch.relu(self.fc2(output))
        output = torch.sigmoid(self.out(output)).squeeze(1)
        return output