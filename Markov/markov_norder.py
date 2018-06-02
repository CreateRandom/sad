import random
import collections

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

    def generate_text(self, char_range, **parameters):
        return self.generate_output(char_range)

    def train(self, file_path, **parameters):
        return self.walk_file(file_path)

    # Generate table
    def generate_table(self, line):
        for word in line.split():
            print(word)
            self.table[tuple(self.seen)].append(word)
            self.seen.append(word)
        self.table[tuple(self.seen)].append(NONWORD)  # Mark end of file

    # table, seen = generate_table("gk_papers.txt")

    # Generate output
    def generate_output(self, char_range):
        self.seen.extend([NONWORD] * self.order)  # clear it all
        toReturn = ''

        char_limit = random.choice(char_range)
        while(len(toReturn) < char_limit):
            word = random.choice(self.table[tuple(self.seen)])
            if word == NONWORD:
                continue
            toReturn = toReturn + ' ' + word
            self.seen.append(word)

        return toReturn

    def walk_file(self, filename):
        lengths = []
        for line in open(filename):
            print(line)
            self.generate_table(line)
            length = len(line)
            if length > 1:
                lengths.append(length)
        return lengths
