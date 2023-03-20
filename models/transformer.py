import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, config, embedding_size, dropout, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size

        den = torch.exp(- torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        token_embedding = token_embedding * math.sqrt(self.embedding_size)
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(self, config, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.out = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = Embedding(config, src_vocab_size, emb_size)
        self.tgt_tok_emb = Embedding(config, tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(config, emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.out(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
