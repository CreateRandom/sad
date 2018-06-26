'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

import os
import pickle
import re

import losswise
import numpy
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

#path = get_file('tweets.txt', origin='https://raw.githubusercontent.com/jpgard/fifty-shades-of-trump/master/tweets.txt')
from keras.wrappers.scikit_learn import KerasRegressor
from losswise.libs import LosswiseKerasCallback
from sklearn.grid_search import GridSearchCV

from LSTM.Utils import downsample, plot_history, perplexity, merge_two_dicts
from Pipeline.Model import Model


class Char_LSTM(Model):

    losswise.set_api_key('EFW1YNI7H')


    def __init__(self, name = None,  **parameters):
        # max length of sentence to use, first n - 1 tokens --> input, last token --> output
        super().__init__(**parameters)
        # learning rate
        self.learning_rate = parameters.get('learning_rate', 0.1)

        self.max_sentence_len = parameters.get('max_sentence_len', 40)
        self.step_size = parameters.get('step_size', 3)
        # percentage of inputs to use for validation
        self.validation_percentage = parameters.get('validation_perc', 0.05)
        self.batch_size = parameters.get('batch_size', 128)
        self.n_epochs = parameters.get('n_epochs', 40)

        if (name is None):
            import time
            self.name = time.strftime("%Y%m%d-%H%M%S")
        else:
            self.name = name

        self.stored_params = parameters


        if not os.path.exists('../logs/' + self.name):
            os.makedirs('../logs/' + self.name)

    def load_data(self, path):
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()
            import re
            self.text = re.sub("[A-Za-z0-9 _.,!?'\"#)(\-$&;:“”’%/…—‘ +–*=\n]+", "", text)
   #         self.text = re.sub("\n", " ", self.text)

        print('corpus length:', len(self.text))
        self.corpus_size = len(self.text)

        chars = sorted(list(set(self.text)))
        self.charset_size = len(chars)

        print('total chars:', self.charset_size)
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.max_sentence_len, self.step_size):
            sentences.append(self.text[i: i + self.max_sentence_len])
            next_chars.append(self.text[i + self.max_sentence_len])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), self.max_sentence_len, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        nb_validation_samples = int(self.validation_percentage * x.shape[0])

        x_train = x[:-nb_validation_samples]
        y_train = y[:-nb_validation_samples]
        x_val = x[-nb_validation_samples:]
        y_val = y[-nb_validation_samples:]

        return x_train, y_train, x_val, y_val


    # build the model: a single LSTM


    def generate_text(self, char_limit, **parameters):
        # Function invoked at end of each epoch. Prints generated text.
        start_index = 1
        while(self.text[start_index] is not "\n"):
            start_index = random.randint(0, self.corpus_size - self.max_sentence_len - 1)
        generated = ''
        sentence = self.text[start_index: start_index + self.max_sentence_len]
        generated += sentence

        while (len(generated) < char_limit):
            x_pred = np.zeros((1, self.max_sentence_len, self.charset_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, 0.1)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        return generated

    def save(self, path):
        self.model.save(path + self.name + '.hd5')

        with open(path + self.name + '.pkl', 'wb') as output:
            pickle.dump(self.corpus_size, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.max_sentence_len, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.charset_size, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.text, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.indices_char, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.char_indices, output, pickle.HIGHEST_PROTOCOL)

    # and loading
    def load(self, model_file):
        self.model = load_model(model_file)
        if model_file.endswith('.hd5'):
            model_file = model_file[:-4]
            model_file = model_file + '.pkl'
        else:
            raise Exception()

        with open(model_file, 'rb') as input:
            self.corpus_size = pickle.load(input)
            self.max_sentence_len = pickle.load(input)
            self.charset_size = pickle.load(input)
            self.text = pickle.load(input)
            self.indices_char = pickle.load(input)
            self.char_indices = pickle.load(input)


    def train(self, data_path, **parameters):
        x_train, y_train, x_val, y_val = self.load_data(data_path)

        def on_epoch_end():
            test = self.generate_text(280)
            print(test)

        def build_model(learning_rate=0.01,n_units=128):
            print('Build model...')
            model = Sequential()
            model.add(LSTM(n_units, input_shape=(self.max_sentence_len, self.charset_size)))
            model.add(Dense(self.charset_size))
            model.add(Activation('softmax'))
            optimizer = RMSprop(lr=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            return model

        best_param = {}
        do_gridsearch = False
        if(do_gridsearch):

            # wrap the model in a scikit-learn regressor
            sci_model = KerasRegressor(build_fn=build_model)

            # define the grid search parameters
            learning_rate = [0.01, 0.05, 0.1]
            n_units = [64,128,256]

            epochs = [5]

            param_grid = dict(learning_rate=learning_rate, epochs=epochs, n_units = n_units)

            # fix random seed for reproducibility
            seed = 42
            numpy.random.seed(seed)
            random.seed(seed)

            # downsample randomly for gridsearch
            if(len(x_train) > 10000):
                train_x_d, train_y_d = downsample(x_train,y_train,10000)
            else:
                train_x_d = x_train
                train_y_d = y_train

            grid = GridSearchCV(estimator=sci_model, param_grid=param_grid, n_jobs=1, verbose=100,cv=3)
            grid_result = grid.fit(train_x_d, train_y_d)
            # summarize results
            best_param = grid_result.best_params_
            print("Best: %f using %s" % (grid_result.best_score_, best_param))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))


        learning_rate = best_param['learning_rate'] if 'learning_rate' in best_param else self.learning_rate
        n_units = best_param['n_units'] if 'n_units' in best_param else 128

        self.model = build_model(learning_rate=learning_rate,n_units=n_units)

        all_params = merge_two_dicts(best_param,self.stored_params)


        # get the training history after fitting the model
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[LambdaCallback(on_epoch_end=on_epoch_end),
                                            LosswiseKerasCallback(params=all_params),
                                            EarlyStopping(patience=2)])

        plot_history(history,self.name)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



model = Char_LSTM(name='Trumpchar')

#model.train('../data/clean_trump.csv')

model.load('../models/Trumpchar.hd5')

for x in range(7):
    print(re.sub("\n", " ", model.generate_text(random.randint(70,140)))[1:])