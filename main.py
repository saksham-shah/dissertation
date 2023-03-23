from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import AttnDecoder
from data import *
from train import *
from config import *
from evaluate import *

output = ""

for dataset in ["mawps", "asdiv"]:
    for rpn in [True, False]:
        for num_emb in [True, False]:
            config["dataset"] = dataset
            config["rpn"] = rpn
            config["num_emb"] = num_emb

            mwps, q_lang, a_lang = load_data(config)

            embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
            encoder = Encoder(config).to(device)
            if config["attention"]:
                decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
            else:
                decoder = Decoder(config, a_lang.n_tokens).to(device)

            train_loader, test_loader = train_test(config, mwps)

            max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 1, print_every=100)

            overall_acc = accuracy(config, mwps, embedding, encoder, decoder, q_lang, a_lang)

            output += "Dataset: " + dataset + "\n"
            output += "RPN: " + ("True\n" if rpn else "False\n")
            output += "Num_emb: " + ("True\n" if num_emb else "False\n")
            output += f"Accuracy - max: {max_acc}, final: {acc}, overall: {overall_acc}, iters: {iters}\n"

print(output)

# torch.save(embedding, 'asdiv-baseline-embedding.pt')
# torch.save(encoder, 'asdiv-baseline-encoder.pt')
# torch.save(decoder, 'asdiv-baseline-decoder.pt')