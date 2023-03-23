from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import AttnDecoder
from data import *
from train import *
from config import *
from evaluate import *

# output = ""

# for dataset in ["mawps", "asdiv"]:
#     for rpn in [True, False]:
#         for num_emb in [True, False]:
#             config["dataset"] = dataset
#             config["rpn"] = rpn
#             config["num_emb"] = num_emb

def run_experiment(config):
    mwps, q_lang, a_lang = load_data(config)

    embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
    encoder = Encoder(config).to(device)
    if config["attention"]:
        decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
    else:
        decoder = Decoder(config, a_lang.n_tokens).to(device)

    train_loader, test_loader = train_test(config, mwps)

    max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=1000)

    # overall_acc = accuracy(config, test_loader, embedding, encoder, decoder, q_lang, a_lang)

    print(acc)

    return max_acc, acc, iters

options = {
    "batch_size": [1, 8],
    "dataset": ["asdiv", "mawps"],
}

args = ["rpn", "num_embs", "batch_size"]

def get_config(base_config, args, exp_num):
    config = base_config.copy()
    for key in args:
        active = exp_num % 2
        exp_num = exp_num >> 1

        if key in options:
            config[key] = options[key][active]
        else:
            config[key] = active == 1
    return config

def run_experiments(base_config, args):
    output = ""
    args = args[::-1]
    for i in range(2 ** len(args)):
        config = get_config(base_config, args, i)
        max_acc, acc, iters = run_experiment(config)

        exp_str = format(i, f'0{len(args)}b')
        exp_output = f"{exp_str} - max acc: {max_acc}, final acc: {acc}, iters: {iters}\n"
        print(exp_output)

        output += exp_output
    print(output)





#             output += "Dataset: " + dataset + "\n"
#             output += "RPN: " + ("True\n" if rpn else "False\n")
#             output += "Num_emb: " + ("True\n" if num_emb else "False\n")
#             output += f"Accuracy - max: {max_acc}, final: {acc}, overall: {overall_acc}, iters: {iters}\n"

# print(output)

# torch.save(embedding, 'asdiv-baseline-embedding.pt')
# torch.save(encoder, 'asdiv-baseline-encoder.pt')
# torch.save(decoder, 'asdiv-baseline-decoder.pt')