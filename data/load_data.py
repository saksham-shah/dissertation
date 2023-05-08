import xml.etree.ElementTree as ET
import json
import re
from torchtext.vocab import GloVe
import numpy as np
from utils.process_input import *
from dataclasses import dataclass
# from config import *

global_vectors = GloVe(name='6B', dim=300)

# mwps = []
# bad_mwps = []
# solution_types = []
# examples = {}

# chars = []

# Q_MAX_LENGTH = 0
# A_MAX_LENGTH = 0

@dataclass
class MWP:
    id: str
    question: str
    equation: str
    answer: float
    numbers: str

def create_MWP(id, question, equation, answer):
    q_tokens, a_tokens, numbers = tokensFromMWP(question, equation)

    if q_tokens is None:
        return None
    
    return MWP(id, " ".join(q_tokens), " ".join(a_tokens), answer, ",".join(map(str, numbers)))

def MWP_from_asdiv(xml_problem):
    id = xml_problem.attrib['ID']
    body = xml_problem.find('Body').text
    question = xml_problem.find('Question').text
    solution_type = xml_problem.find('Solution-Type').text
    answer = xml_problem.find('Answer').text
    formula = xml_problem.find('Formula').text

    if solution_type not in ['Addition', 'Subtraction', 'Multiplication', 'Common-Division', 'Sum', 'Difference', 'TVQ-Initial', 'TVQ-Change', 'TVQ-Final']:
        # print(solution_type)
        # print(body + question)
        # print(formula + "<>" + answer)
        return None
        
    solution_attrib = xml_problem.find('Solution-Type').attrib
    if 'UnitTrans' in solution_attrib:
        # print(body + question)
        # print(formula)
        return None
    
    id = "asdiv" + str(id.split("-")[1])

    equation = formula.split('=')[0]
    answer = float(re.sub(r" \(.+\)", r"", answer))

    question = body + question
    question = re.sub(r"([?$])", r" \1 ", question)
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question)

    return create_MWP(id, question, equation, answer)

def MWP_from_mawps(mawps):
    id = "mawps" + str(mawps["id"])
    question = mawps["original_text"]
    answer = float(mawps["ans"])

    if mawps["equation"].lower()[:2] != 'x=':
        return None

    equation = mawps["equation"].split('=')[1]

    # print(question, equation, answer)

    return create_MWP(id, question, equation, answer)

def MWP_from_svamp(svamp):
    id = "svamp" + str(svamp["ID"].split("-")[1])
    question = svamp["Body"] + svamp["Question"]
    question = re.sub(r"([?$])", r" \1 ", question)
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question)

    answer = float(svamp["Answer"])
    equation = svamp["Equation"]

    return create_MWP(id, question, equation, answer)

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: 'SOS', 1: 'EOS'}
        self.n_tokens = 2
        self.weights = None

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
    
    def set_weights(self, weights):
        self.weights = weights

def load_data():
    mwps = {}
    ids = []

    # if config["dataset"] in ["asdiv", "both"]:
    with open('./data/asdiv/asdiv_a.txt') as file:
        asdiv_a_ids = [line.rstrip() for line in file]
    
    tree = ET.parse('./data/asdiv/asdiv.xml')
    root = tree.getroot()

    for child in root:
        if child.attrib['ID'] in asdiv_a_ids:
            mwp = MWP_from_asdiv(child)
            if mwp is not None:
                mwps[mwp.id] = mwp
                ids.append(mwp.id)
            # else:
            #     print(child.find('Body').text + child.find('Question').text)
            #     print(child.find('Solution-Type').text)
            #     print(child.find('Formula').text + child.find('Answer').text)
    
    # if config["dataset"] in ["mawps", "both"]:
    with open('./data/mawps/mawps.json') as file:
        mawps = json.loads(file.read())

    for mawps_q in mawps:
        mwp = MWP_from_mawps(mawps_q)
        if mwp is not None:
            mwps[mwp.id] = mwp
            ids.append(mwp.id)

    return mwps, ids

def generate_vocab(config, mwps):
    q_lang = Lang()
    a_lang = Lang()

    tokens = {}
    for mwp in mwps:
        for token in mwp.question.split(" "):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    
    for mwp in mwps:
        unk_updated = []
        for token in mwp.question.split(" "):
            if tokens[token] > 1 or token[0] == '#':
                unk_updated.append(token)
            else:
                unk_updated.append("UNK")
        mwp.question = " ".join(unk_updated)

    for mwp in mwps:
        # q_tokens, a_tokens, _ = tokensFromMWP(mwp.question, mwp.equation)

        q_lang.addTokens(mwp.question.split(" "))
        a_lang.addTokens(mwp.equation.split(" "))

        # if (len(mwp.question) > Q_MAX_LENGTH):
        #     Q_MAX_LENGTH = len(mwp.question)
        # if (len(mwp.equation) > A_MAX_LENGTH):
        #     A_MAX_LENGTH = len(mwp.equation)

    if "embedding_size" in config:
        embedding_size = config["embedding_size"]

        q_weights_matrix = np.zeros((q_lang.n_tokens, embedding_size))
        a_weights_matrix = np.zeros((a_lang.n_tokens, embedding_size))

        for i, token in enumerate(q_lang.token2index):
            embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
            if embedding.norm() != 0:
                q_weights_matrix[i] = embedding
            else:
                q_weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size, ))

        for i, token in enumerate(a_lang.token2index):
            embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
            if embedding.norm() != 0:
                a_weights_matrix[i] = embedding
            else:
                a_weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size, ))
        
        q_lang.set_weights(q_weights_matrix)
        a_lang.set_weights(a_weights_matrix)
    
    return q_lang, a_lang