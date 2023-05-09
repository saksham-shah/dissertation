import random
from data.load_data import *
from utils.load_batches import *
import torch
from models import Classifier, AttnClassifier
from utils.prepare_tensors import *
from timeit import default_timer as timer

config = {
    "num_layers": 1,
    "batch_size": 1,
    "teacher_forcing_ratio": 0.9,
    "learning_rate": 0.0005,
    "hidden_size": 256,
    "bidirectional": True,
    "dropout": 0.1,
    "early_stopping": 5,
    "rpn": False,
    "num_emb": True,
    "embedding_size": 300,
    "dataset": ["asdiv", "mawps"],
    "attention": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Return list of all possible equations
# 2 numbers and one operator between them, e.g. #1 + #0
# Use RPN for convenience
def get_all_equations():
    max_numbers = 3 # i.e. #0, #1, #2
    operations = ["+", "-", "*", "/"]

    all_pairs = []

    for i in range(max_numbers):
        for j in range(max_numbers):
            if i == j:
                continue
            all_pairs.append(f"#{str(i)} #{str(j)} ")

    all_equations = []

    for pair in all_pairs:
        for operation in operations:
            all_equations.append(pair + operation)

    return all_equations

# Check if two equations are equivalent
# e.g. #0 + #1 and #1 + #0
def is_equivalent(e1, e2):
    if e1 == e2:
        return True
    
    tokens1 = e1.split(" ")
    tokens2 = e2.split(" ")

    if len(tokens1) != len(tokens2):
        return False

    # Check if operators are the same
    # Assume RPN (e.g. "#0 #1 +") so use index 2
    if tokens1[2] != tokens2[2]:
        return False
    
    # Commutativity of addition/multiplication
    if tokens1[2] in ["+", "*"]:
        return (tokens1[0] == tokens2[1]) and (tokens2[0] == tokens1[1])

all_equations = get_all_equations()

# Evaluate model accuracy on test set
def evaluate(config, model, test_loader, q_lang, a_lang):
    correct = 0

    for mwp in test_loader:
        q_tensor = tensorFromTokens(q_lang.token2index, mwp['question'][0].split(" ")).view(-1, 1)
        numbers = list(map(float, mwp['numbers'][0].split(",")))

        max_output = 0
        best_equation = None

        # Get match confidence for all possible equations
        for equation in all_equations:
            a_tensor = tensorFromTokens(a_lang.token2index, equation.split(" ")).view(-1, 1)
            output = model(q_tensor, a_tensor, [q_tensor.size(0)], [a_tensor.size(0)], [numbers])

            # Select equation with highest confidence
            if output > max_output:
                max_output = output
                best_equation = equation
        
        # Correct if selected equation is equivalent to target equation
        if is_equivalent(best_equation, mwp['formula'][0]):
            correct += 1
            
    return correct / len(list(test_loader))

# Train model on train set
# Return average training loss
def train(config, model, train_loader, optimiser, criterion, q_lang, a_lang):
    model.train()

    losses = 0

    force_correct = 0.5 # proportion of matches in training data

    for mwp in train_loader:
        # Prepare question tensor
        q_indexes = [indexesFromTokens(q_lang.token2index, q.split(" ")) for q in mwp['question']]
        q_lengths = [len(q) for q in q_indexes]
        q_padded = [pad_indexes(q, max(q_lengths)) for q in q_indexes]
        q_tensor = torch.tensor(q_padded, device=device).transpose(0, 1)

        numbers = [list(map(float, nums.split(","))) for nums in mwp['numbers']]

        # Test on all possible equations
        for equation in all_equations:
            # Ensure equal split between matches and mismatches
            a_strings = [a if random.random() < force_correct else equation for a in mwp['formula']]
            target = [1.0 if is_equivalent(a, equation) else 0.0 for a in a_strings]
            
            # Prepare equation (answer) tensor
            a_indexes = [indexesFromTokens(a_lang.token2index, a.split(" ")) for a in a_strings]
            a_lengths = [len(a) for a in a_indexes]
            a_padded = [pad_indexes(a, max(a_lengths)) for a in a_indexes]
            a_tensor = torch.tensor(a_padded, device=device).transpose(0, 1)

            output = model(q_tensor, a_tensor, q_lengths, a_lengths, numbers)

            optimiser.zero_grad()

            loss = criterion(output, torch.tensor(target, device=device))
            loss.backward()

            optimiser.step()
            losses += loss.item()

    return losses / len(list(train_loader)) / len(all_equations)

# Train classifier for specified number of epochs
def train_classifier(config, classifier, train_loader, test_loader, q_lang, a_lang, epochs=50):
    optimiser = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCELoss() # binary cross entropy as we use binary classifier

    for epoch in range(1, epochs+1):
        start_time = timer()
        train_loss = train(config, classifier, train_loader, optimiser, criterion, q_lang, a_lang)
        val_loss = evaluate(config, classifier, test_loader, q_lang, a_lang)
        end_time = timer()
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val acc: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))