from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from train import *
from config import *
from evaluate import *

output = ""

for rpn in [True, False]:
    for num_emb in [True, False]:
        config["rpn"] = rpn
        config["num_emb"] = num_emb

        embedding = Embedding(config, q_lang.n_tokens, EMBEDDING_SIZE, q_lang, q_weights_matrix).to(device)
        encoder = Encoder(config, EMBEDDING_SIZE).to(device)
        attn_decoder = Decoder(config, a_lang.n_tokens, EMBEDDING_SIZE).to(device)

        max_acc, acc, iters = trainIters(config, embedding, encoder, attn_decoder, 50, print_every=100)

        overall_acc = accuracy(config, embedding, encoder, attn_decoder)

        output += "RPN: " + ("True\n" if rpn else "False\n")
        output += "Num_emb: " + ("True\n" if num_emb else "False\n")
        output += f"Accuracy - max: {max_acc}, final: {acc}, overall: {overall_acc}, iters: {iters}\n"

print(output)

# torch.save(embedding, 'asdiv-baseline-embedding.pt')
# torch.save(encoder, 'asdiv-baseline-encoder.pt')
# torch.save(attn_decoder, 'asdiv-baseline-attndecoder.pt')