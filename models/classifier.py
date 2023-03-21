import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNClassifier(nn.Module):
    def __init__(self, config, hidden_1, hidden_2):
        super(RNNClassifier, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.bidirectional = config["bidirectional"]

        self.q_encoder = Encoder(config)
        self.a_encoder = Encoder(config)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, q_embedded, q_input_lengths, q_restore_indexes, a_embedded, a_input_lengths, a_restore_indexes, hidden=None):
        q_output, _ = self.q_encoder(q_embedded, q_input_lengths, q_restore_indexes)
        a_output, _ = self.a_encoder(a_embedded, a_input_lengths, a_restore_indexes)

        output = torch.concat([q_output[0], a_output[0]], dim=0)

        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output = torch.sigmoid(self.out(output))
        return output