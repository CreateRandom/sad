import random
import collections
import os

# Since we split on whitespace, this can never be a word
from Pipeline.Model import Model

NONWORD = "\n"


class Markov(Model):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        # get the order parameter, default value is 2
        self.order = parameters.get('order', 2)

        self.table = collections.defaultdict(list)
        self.seen = collections.deque([NONWORD] * self.order, self.order)

    def generate_text(self, char_limit, **parameters):
        return self.generate_output(char_limit)

    def train(self, data_path, **parameters):
        self.walk_directory(data_path)

    # Generate table
    def generate_table(self, filename):
        for line in open(filename):
            for word in line.split():
                self.table[tuple(self.seen)].append(word)
                self.seen.append(word)
        self.table[tuple(self.seen)].append(NONWORD)  # Mark end of file

    # table, seen = generate_table("gk_papers.txt")

    # Generate output
    def generate_output(self, char_limit=140):
        self.seen.extend([NONWORD] * self.order)  # clear it all
        toReturn = ''

        while(len(toReturn) < char_limit):
            word = random.choice(self.table[tuple(self.seen)])
            if word == NONWORD:
                continue
            toReturn = toReturn + ' ' + word
            self.seen.append(word)

        return toReturn

    def walk_directory(self, rootDir):
        for dirName, subdirList, fileList in os.walk(rootDir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                self.generate_table(os.path.join(dirName, fname))
            # print('\t%s' % fname)
