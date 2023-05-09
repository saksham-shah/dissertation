import math
import random
import torch
import transformers
import numpy as np
from functools import partial
print(transformers.__version__)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from evaluation.experiment import *

model_checkpoint = "facebook/bart-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "rpn": True,
    "dataset": ['asdiv', 'mawps'],
    "epochs": 50,
    "weight_decay": 0.01,
}

# HuggingFace uses dict rather than dataclass object
def mwp_to_dict(mwp):
    return {
        "id": mwp.id,
        "question": mwp.question,
        "equation": mwp.equation,
        "answer": mwp.answer,
        "numbers": mwp.numbers,
    }

# Split by testing proportion
def train_test_split(data, test_size=0.1):
    random.seed(1)
    random.shuffle(data)

    boundary = math.floor(len(data) * (1 - test_size))
    train = data[:boundary]
    test = data[boundary:]
    return { "train": train, "test": test }

# Specific format for dataset: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
class MWPDataset(torch.utils.data.Dataset):
  def __init__(self, inputs, targets):
    self.inputs = inputs
    self.targets = targets

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
    item['labels'] = torch.tensor(self.targets['input_ids'][idx])
    return item
  
  def __len__(self):
    return len(self.inputs['input_ids'])

# Load and split data
def get_data(config):
    mwps = prepare_training_data(config['dataset'])
    print(f"Num mwps: {len(mwps)}")
    data = list(map(mwp_to_dict, mwps))

    inputs = train_test_split([mwp["question"] for mwp in data])
    targets = train_test_split([mwp["equation"] for mwp in data])
    mwps = train_test_split(data)

    return inputs, targets, mwps

# Tokenise using transformers.AutoTokenizer
def tokenise_data(tokeniser, inputs, targets):
    max_input_length = 1024
    max_target_length = 64

    tokenised_inputs = {
        "train": tokeniser(inputs["train"], max_length=max_input_length, truncation=True),
        "test": tokeniser(inputs["test"], max_length=max_input_length, truncation=True),
    }

    tokenised_targets = {
        "train": tokeniser(targets["train"], max_length=max_target_length, truncation=True),
        "test": tokeniser(targets["test"], max_length=max_target_length, truncation=True),
    }

    train_dataset = MWPDataset(tokenised_inputs["train"], tokenised_targets["train"])
    test_dataset = MWPDataset(tokenised_inputs["test"], tokenised_targets["test"])

    return train_dataset, test_dataset

# Fine-tune BART using Seq2SeqTrainer
def train_model(config, model, tokeniser, train_dataset, test_dataset, test_mwps):
    batch_size = 16
    args = Seq2SeqTrainingArguments(
        f"{model_checkpoint}-finetunes-mawps",
        evaluation_strategy = "epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=50,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    data_collator = DataCollatorForSeq2Seq(tokeniser, model=model)

    # Compute custom evaluation metric
    def compute_metrics(tokeniser, mwps, eval_pred):
        correct = 0

        for i in range(len(eval_pred.predictions)):
            mwp = mwps[i]

            numbers = list(map(float, mwp['numbers'].split(",")))
            answer = mwp["answer"]

            pred_tokens = np.expand_dims(eval_pred.predictions[i], 0)
            pred = [tokeniser.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=False) for token in pred_tokens]
            
            # Same evaluation as for seq2seq model
            rpn_exp = infix_to_rpn(pred[0].split(" "))
            output_ans = eval_rpn(rpn_exp, numbers)

            if output_ans is not None and math.isclose(output_ans, answer, rel_tol=1e-4):
                correct += 1
            
        return {
            'accuracy': correct / len(eval_pred.predictions)
        }

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokeniser,
        compute_metrics=partial(compute_metrics, tokeniser, test_mwps),
    )

    print("Training now...")

    trainer.train()

    return trainer

# Determine if model is correct for single example
def is_correct(model, input_tokens, target, numbers, answer, tokeniser, attempts=3):
    for _ in range(attempts):
        # Decode tokens into text using AutoTokeniser
        pred_tokens = model.generate(input_tokens['input_ids'].to(device), num_beams=4, max_length=32, early_stopping=True)
        pred = [tokeniser.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=False) for token in pred_tokens]

        rpn_exp = infix_to_rpn(pred[0].split(" "))
        output_ans = eval_rpn(rpn_exp, numbers)

        if output_ans is None:
            print("RETRYING")
            continue

        if math.isclose(output_ans, answer, rel_tol=1e-4):
            print("CORRECT:", pred[0], "<>", target)
            return True
        else:
            print("WRONG:", pred[0], " @@@ ", target)
            return False
    return False

# Evaluate and return model accuracy
def evaluate_accuracy(model, tokeniser, inputs, targets, mwps):
    print("Evaluating...")
    correct = 0
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        mwp = mwps[i]

        numbers = list(map(float, mwp['numbers'].split(",")))
        answer = mwp["answer"]

        input_tokens = tokeniser([input], max_length=1024, return_tensors='pt')
        input_tokens['input_ids']#.to(device)

        if is_correct(model, input_tokens, target, numbers, answer, tokeniser):
            correct += 1

    return correct / len(inputs)