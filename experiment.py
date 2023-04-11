from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import AttnDecoder
from data import *
from train import *
from evaluate import *

import numpy as np

def run_experiment(config, mwps, nfold=10):
    overall_acc = 0

    accs = []

    for fold in range(nfold):
        train_set, test_set = train_test(mwps, fold=fold, nfold=nfold)

        q_lang, a_lang = generate_vocab(config, train_set)

        train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
        test_loader = batch_data(test_set, config['rpn'], 1)

        embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
        encoder = Encoder(config).to(device)
        if config["attention"]:
            decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
        else:
            decoder = Decoder(config, a_lang.n_tokens).to(device)

        max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=0)
        print(f"Fold: {fold}, Accuracy: {max_acc}")

        overall_acc += max_acc
        accs.append(max_acc)

    return overall_acc / nfold, accs

def get_config(base_config, args, exp_num):
    options = {
        "batch_size": [1, 4],
        "dataset": ["asdiv", "mawps"],
    }
    
    config = base_config.copy()
    for key in args:
        active = exp_num % 2
        exp_num = exp_num >> 1

        if key in options:
            config[key] = options[key][active]
        else:
            config[key] = active == 1
    return config

def run_experiments(base_config, args, mwps, nfold=10):
    output = ""
    args = args[::-1]
    folds = []
    folds_avg = []
    for i in range(2 ** len(args)):
        config = get_config(base_config, args, i)
        acc, fold_accs = run_experiment(config, mwps, nfold=nfold)
        folds.append(fold_accs)
        folds_avg.append(acc)

        exp_str = format(i, f'0{len(args)}b')
        exp_output = f"{exp_str} - avg acc: {acc}\n"
        print(exp_output)

        output += exp_output
    print(output)

    return folds_avg, folds

def prepare_training_data(datasets=['asdiv', 'mawps']):
    all_mwps, ids = load_data()

    with open('data/test.txt') as file:
        test_ids = [line.rstrip() for line in file]

    mwps = []
    for mwp in all_mwps:
        if mwp not in test_ids:
            if mwp[:5] in datasets:
                mwps.append(all_mwps[mwp])

    random.shuffle(mwps)
    return mwps

t_values = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812]

def paired_t_test(acc, t_value=None):
    n_folds = len(acc[0])
    if t_value is None:
        t_value = t_values[n_folds - 1]

    acc = np.array(acc)

    # diff: n_models x n_models x n_folds
    diff = acc[:, np.newaxis, :] - acc

    # mean: n_models x n_models
    mean = np.mean(diff, 2)

    sd = np.std(diff, 2, ddof=1)

    t_stat = np.divide(np.sqrt(n_folds) * mean, sd, where=sd!=0)

    return t_stat, t_stat > t_value