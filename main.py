from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from train import *
from config import *
from evaluate import *

embedding = Embedding(config, q_lang.n_tokens, EMBEDDING_SIZE, q_weights_matrix).to(device)
encoder = Encoder(config, EMBEDDING_SIZE).to(device)
attn_decoder = Decoder(config, a_lang.n_tokens, EMBEDDING_SIZE).to(device)

trainIters(config, embedding, encoder, attn_decoder, 50, print_every=100)

accuracy(embedding, encoder, attn_decoder)

torch.save(embedding, 'asdiv-baseline-embedding.pt')
torch.save(encoder, 'asdiv-baseline-encoder.pt')
torch.save(attn_decoder, 'asdiv-baseline-attndecoder.pt')