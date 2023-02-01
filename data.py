import xml.etree.ElementTree as ET
import re
from word2number import w2n
from torchtext.vocab import GloVe
import numpy as np

global_vectors = GloVe(name='6B', dim=300)

tree = ET.parse('data\\asdiv.xml')
root = tree.getroot()

mwps = {}
solution_types = []
examples = {}

chars = []

with open('data\\asdiv_a.txt') as file:
    asdiv_a_ids = [line.rstrip() for line in file]

def tokenise_question(s):
    s = s.lower()
    s = re.sub(r"([?$])", r" \1 ", s)
    # s = re.sub(r" ?([.,]) ?", r" \1 ", s)
    s = re.sub(r"([.,])([^0-9])", r" \1 \2", s)
    # s = re.sub(r"([.,])([^0-9])", r" \1 \2", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def tokenise_formula(s):
    s = re.sub(r"([()+\-*/=])", r" \1 ", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def string_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            return float(w2n.word_to_num(s))
        except ValueError:
            if "," in s and len(s) > 1:
                return string_to_float(re.sub(r",", r"", s))
            return None

class MWP:
    def __init__(self, xml_problem):
        self.body = xml_problem.find('Body').text
        self.question = xml_problem.find('Question').text
        self.solution_type = xml_problem.find('Solution-Type').text
        self.answer = xml_problem.find('Answer').text
        self.formula = xml_problem.find('Formula').text

        self.valid = self.solution_type in ['Addition', 'Subtraction', 'Multiplication', 'Common-Division']
        solution_attrib = xml_problem.find('Solution-Type').attrib
        if 'UnitTrans' in solution_attrib:
            self.valid = False
            print("translation", self.body)

        self.numbers = []

        self.full_question = self.body + self.question

        self.target = self.formula.split('=')[0]

        self.q_tokens = tokenise_question(self.full_question)
        self.a_tokens = tokenise_formula(self.target)

        self.answer = float(re.sub(r" \(.+\)", r"", self.answer))

        if self.valid:
            # try:
            #     if float(self.split[1]) != self.answer:
            #         print("ASDASDASDASD", self.split[1], self.answer)
            # except ValueError:
            #     print(self.solution_type)
            #     print(self.split[1], self.answer)
                

            for i in range(len(self.q_tokens)):
                num = string_to_float(self.q_tokens[i])
                if num is not None:
                    if num not in self.numbers:
                        self.numbers.append(num)
                        self.q_tokens[i] = "#" + str(len(self.numbers) - 1)
                    else:
                        print(self.q_tokens)
            
            for i in range(len(self.a_tokens)):
                num = string_to_float(self.a_tokens[i])
                if num is not None and num in self.numbers:
                        self.a_tokens[i] = "#" + str(self.numbers.index(num))

            # if len(self.numbers) != 2:
            #     self.valid = False

        for char in self.full_question:
            if char not in chars:
                chars.append(char)
            
            if char == ':':
                print(self.full_question)

        # if self.valid:
        #     if self.solution_type not in solution_types:
        #         solution_types.append(self.solution_type)
        #         examples[self.solution_type] = self                

for child in root:
    if child.attrib['ID'] in asdiv_a_ids:
        mwps[child.attrib['ID']] = MWP(child)

def analyse_formula(formula, answer):
    split_formula = formula.split('=')
    # print(split_formula)

    # answer = re.sub(r" \(.+\)", r"", answer)

    if (float(split_formula[1]) != answer):
        print(formula, "<"+answer+">", "<"+split_formula[1]+">")

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

valid_mwps = []

count = 0
Q_MAX_LENGTH = 0
A_MAX_LENGTH = 0
for key in mwps:
    if mwps[key].valid:
        count += 1

        q_lang.addTokens(mwps[key].q_tokens)
        a_lang.addTokens(mwps[key].a_tokens)

        valid_mwps.append(mwps[key])

        if (len(mwps[key].q_tokens) > Q_MAX_LENGTH):
            Q_MAX_LENGTH = len(mwps[key].q_tokens)
        if (len(mwps[key].a_tokens) > A_MAX_LENGTH):
            A_MAX_LENGTH = len(mwps[key].q_tokens)

Q_MAX_LENGTH = 100
A_MAX_LENGTH = 30

        # for token in mwps[key].q_tokens:
        #     if (',' in token or '.' in token) and len(token) > 1:
            # if token == '':
            #     print(token)
            #     print(mwps[key].full_question)

EMBEDDING_SIZE = 300

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

print(count)

print("".join(chars))

print(q_lang.n_tokens)
print(a_lang.token2count)

# REMOVED SYMBOLS: = ( ) & " : CHECK FOR MORE LATER