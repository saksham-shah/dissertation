from models import *
from data.load_data import *
from seq2seq import *

import numpy as np

# Train a single model configuration
# Returns average fold accuracy and list of accuracies for each fold
def run_experiment(config, mwps, nfold=10):
    overall_acc = 0

    accs = []

    for fold in range(nfold):
        # Split data by fold number and generate vocab for this fold
        train_set, test_set = train_test(mwps, fold=fold, nfold=nfold)

        q_lang, a_lang = generate_vocab(config, train_set)

        # Batch data into data_loader
        train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
        test_loader = batch_data(test_set, config['rpn'], 1)

        # Define PyTorch models
        embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
        encoder = Encoder(config).to(device)
        if config["attention"]:
            decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
        else:
            decoder = Decoder(config, a_lang.n_tokens).to(device)

        # Train model
        max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=0)
        print(f"Fold: {fold}, Accuracy: {max_acc}")

        overall_acc += max_acc
        accs.append(max_acc)

    return overall_acc / nfold, accs

# Get config object by experiment number
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

# General case of above function for non-binary options
def get_config_general(base_config, args, exp_num):
    config = base_config.copy()

    for key in args:
        num_options = len(args[key])
        index = exp_num % num_options
        exp_num = math.floor(exp_num / num_options)
        config[key] = args[key][index]

    return config

# Run experiments varying specific options (in args)
# Returns all fold accuracies for all configurations
def run_experiments(base_config, args, mwps, nfold=10, binary=True):
    output = ""
    folds = []
    folds_avg = []
    if binary: # all options are binary (i.e. whether or not attention is used)
        args = args[::-1]
        num_exps = 2 ** len(args)
    else: # general case for non-binary options (e.g. batch size)
        args = dict(reversed(list(args.items())))
        num_exps = 1
        for key in args:
            num_exps *= len(args[key])

    for i in range(num_exps):
        if binary:
            config = get_config(base_config, args, i)
        else:
            config = get_config_general(base_config, args, i)

        acc, fold_accs = run_experiment(config, mwps, nfold=nfold)
        folds.append(fold_accs)
        folds_avg.append(acc)

        exp_str = format(i, f'0{len(args)}b')
        exp_output = f"{exp_str} - avg acc: {acc}\n"
        print(exp_output)

        output += exp_output
    print(output)

    return folds_avg, folds

# Load and filter training data
def prepare_training_data(datasets=['asdiv', 'mawps']):
    all_mwps, ids = load_data()

    with open('data/test.txt') as file:
        test_ids = [line.rstrip() for line in file]

    mwps = []
    for mwp in all_mwps:
        if mwp not in test_ids: # filter out held-out MWPs
            if mwp[:5] in datasets:
                mwps.append(all_mwps[mwp])

    random.shuffle(mwps)
    return mwps

# t_values for p = 0.05 for different degrees of freedom
t_values = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812]

# Compute result of paired t-test on matrix of fold accuracies
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