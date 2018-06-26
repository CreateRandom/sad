import random
import collections
from spacy.lang.en import English
from Pipeline.Model import Model

START = "^^"
END = "$$"
parser = English()


class Markov(Model):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        # get the order parameter, default value is 2
        self.order = parameters.get('order', 2)

        self.table = collections.defaultdict(list)
        self.seen = collections.deque([START] * self.order, self.order)

    # Main generate function
    def generate_text(self, char_limit, file_path, **parameters):
        return self.generate_output(char_limit, file_path)

    # Check that generated sentence not exact replica
    def replica(self, sentence, file_path):
        trainFile = open(file_path)
        text = trainFile.read().replace('\n', ' ').lower()
        if sentence in text:
            return True
        return False

    # Takes a list of strings and creates a string from it
    def listToString(self, inputList):
        # paste together special characters in words with previous word
        pre = ['!', '%', ')', '*', ',', '.', '/', '\'', "'",
               '>', '?', '\\', ']', '^', '_', '`', '|',
               '}', '~']
        post = ['#', '$', '[', '(', '@', '<', '{']
        outputList = []
        skipNext = False
        for idx in range(len(inputList)):
            appended = False
            if skipNext:
                skipNext = False
                continue
            for char in inputList[idx]:
                if char in pre:
                    if idx > 0:
                        concat = outputList[-1] + inputList[idx]
                        outputList = outputList[:-1]
                    else:
                        concat = inputList[-1] + inputList[idx]
                    outputList.append(concat)
                    appended = True

                if char in post:
                    if idx == len(inputList) - 1:
                        outputList.append(inputList[idx])
                    else:
                        outputList.append(inputList[idx] + inputList[idx+1])
                        skipNext = True
                    appended = True
            if not appended:
                outputList.append(inputList[idx])
        output = " ".join(outputList)
        return output

    # Main train function
    def train(self, file_path, **parameters):
        return self.train_on_file(file_path)

    # Generate output
    def generate_output(self, char_limit, file_path):
        self.seen.extend([START] * self.order)  # clear it all
        outputLen = 0
        outputList = []

        while outputLen < char_limit:
            word = random.choice(self.table[tuple(self.seen)])
            if word == END:
                break
            outputLen += len(word) + 1
            outputList.append(word)
            self.seen.append(word)

        output = self.listToString(outputList)

        if self.replica(output, file_path):
            return self.generate_output(char_limit, file_path)
        else:
            return output

    # Train on a certain file containing tweet text
    def train_on_file(self, filename):
        lengths = []
        for line in open(filename):
            self.generate_table(line)
            length = len(line)
            if length > 1:
                lengths.append(length)
        return lengths

    # Generate table from tweets
    def generate_table(self, line):
        tokens = parser(line)
        tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
        self.seen = collections.deque([START] * self.order, self.order)
        for word in tokens:
            word = word.lower()
            self.table[tuple(self.seen)].append(word)
            self.seen.append(word)
        self.table[tuple(self.seen)].append(END)
