import xml.etree.ElementTree as ET
import json
import re
from word2number import w2n
from torchtext.vocab import GloVe
import numpy as np
from utils.process_input import *
from config import *

global_vectors = GloVe(name='6B', dim=300)

mwps = []
bad_mwps = []
solution_types = []
examples = {}

chars = []

class MWP:
    def __init__(self, question, equation, answer):
        q_tokens, a_tokens, numbers = tokensFromMWP(question, equation, rpn=config["rpn"])

        if q_tokens is None:
            bad_mwps.append(question + " " + equation)
            self.valid = False
            return

        self.valid = True
        self.question = " ".join(q_tokens)
        self.equation = " ".join(a_tokens)
        self.answer = answer
        self.numbers = ",".join(map(str, numbers))

def MWP_from_asdiv(xml_problem):
    body = xml_problem.find('Body').text
    question = xml_problem.find('Question').text
    solution_type = xml_problem.find('Solution-Type').text
    answer = xml_problem.find('Answer').text
    formula = xml_problem.find('Formula').text

    if solution_type  not in ['Addition', 'Subtraction', 'Multiplication', 'Common-Division', 'Sum', 'Difference', 'TVQ-Initial', 'TVQ-Change', 'TVQ-Final']:
        return None
    
    solution_attrib = xml_problem.find('Solution-Type').attrib
    if 'UnitTrans' in solution_attrib:
        print(body + question)
        print(formula)
        return None

    equation = formula.split('=')[0]
    answer = float(re.sub(r" \(.+\)", r"", answer))

    question = body + question
    question = re.sub(r"([?$])", r" \1 ", question)
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question)

    return MWP(question, equation, answer)

def MWP_from_mawps(mawps):
    question = mawps["original_text"]
    answer = float(mawps["ans"])
    equation = mawps["equation"].split('=')[1]

    # print(question, equation, answer)

    return MWP(question, equation, answer)
    
if config["dataset"] == "asdiv":
    with open('data/asdiv_a.txt') as file:
        asdiv_a_ids = [line.rstrip() for line in file]
    
    tree = ET.parse('data/asdiv.xml')
    root = tree.getroot()

    for child in root:
        if child.attrib['ID'] in asdiv_a_ids:
            mwp = MWP_from_asdiv(child)
            if mwp is not None and mwp.valid:
                mwps.append(mwp)
elif config["dataset"] == "mawps":
    with open('data/mawps.json') as file:
        mawps = json.loads(file.read())
        for mawps_q in mawps:
            mwp = MWP_from_mawps(mawps_q)
            if mwp is not None and mwp.valid:
                mwps.append(mwp)

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: 'SOS', 1: 'EOS'}
        self.n_tokens = 2

    def addTokens(self, tokens):
        for token in tokens:
            self.addToken(token)
    
    def addToken(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

q_lang = Lang()
a_lang = Lang()

for mwp in mwps:
    # q_tokens, a_tokens, _ = tokensFromMWP(mwp.question, mwp.equation)

    q_lang.addTokens(mwp.question.split(" "))
    a_lang.addTokens(mwp.equation.split(" "))

valid_mwps = mwps

EMBEDDING_SIZE = config["embedding_size"]

q_weights_matrix = np.zeros((q_lang.n_tokens, EMBEDDING_SIZE))
a_weights_matrix = np.zeros((a_lang.n_tokens, EMBEDDING_SIZE))

for i, token in enumerate(q_lang.token2index):
  embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
  if embedding.norm() != 0:
    q_weights_matrix[i] = embedding
  else:
    q_weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_SIZE, ))

for i, token in enumerate(a_lang.token2index):
  embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
  if embedding.norm() != 0:
    a_weights_matrix[i] = embedding
  else:
    a_weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_SIZE, ))