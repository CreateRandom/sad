import random
import collections
from spacy.lang.en import English
from Pipeline.Model import Model

START = "^"
END = "$"
parser = English()


class Markov(Model):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        # get the order parameter, default value is 2
        self.order = parameters.get('order', 2)

        self.table = collections.defaultdict(list)
        self.seen = collections.deque([START] * self.order, self.order)

    # Main generate function
    def generate_text(self, char_limit, **parameters):
        return self.generate_output(char_limit)

    # Main train function
    def train(self, file_path, **parameters):
        return self.train_on_file(file_path)

    # Generate output
    def generate_output(self, char_limit):
        self.seen.extend([START] * self.order)  # clear it all
        output = ''

        while len(output) < char_limit:
            word = random.choice(self.table[tuple(self.seen)])
            if word == END:
                break
            output = output + ' ' + word  # TODO more sophisticated?
            self.seen.append(word)
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
            self.table[tuple(self.seen)].append(word)
            self.seen.append(word)
        self.table[tuple(self.seen)].append(END)
