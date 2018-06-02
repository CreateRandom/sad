import os
import sys

from keras import Sequential
from keras.callbacks import LambdaCallback
# a sci-kit learn wrapper for Keras classifier
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

from Pipeline.Model import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Activation

import numpy as np
# compare https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
class GLove_LSTM(Model):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.num_words_to_keep = parameters.get('num_words_to_keep',10000)
        self.num_words_to_keep = parameters.get('num_words_to_keep',10000)
        self.max_sentence_len = parameters.get('max_sentence_len',40)
        self.validation_percentage = parameters.get('validation_perc',0.05)

        self.n_epochs = parameters.get('n_epochs', 10)
        self.batch_size = parameters.get('batch_size', 128)

        if not 'glove_embedding_path' in parameters:
            raise Exception('For running the gLove model, please provide the path of the embedding file'
                            'as a parameter.')

        self.embedding_path = parameters.get('glove_embedding_path', None)

        # TODO set this automatically based on the embedding that was read in
        self.embedding_size = parameters.get('embedding_size',50)

        self.word_to_index = None

    def train(self, data_path, **parameters):

        # train samples and validation set
        train_x, train_y, val_x, val_y = self.import_data(data_path)

        embeddings_index = self.import_embeddings(self.embedding_path)

        vocab_size = len(self.word_to_index) + 1

        # generate the embedding matrix
        embedding_matrix = np.zeros((vocab_size, self.embedding_size))
        for word, i in self.word_to_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector


        # load this into a layer for mapping the data

        embedding_layer = Embedding(len(self.word_to_index) + 1,
                                    self.embedding_size,
                                    weights=[embedding_matrix],
                                    trainable=False)

        print('\nTraining LSTM...')

        def build_model():
            model = Sequential()
            model.add(embedding_layer)
            model.add(LSTM(units=self.embedding_size))
            model.add(Dense(units=vocab_size))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            return model

        def on_epoch_end(epoch, _):
            print('\nGenerating text after epoch: %d' % epoch)
            texts = [
                "I'll be in", 'Marco Rubio is', 'The media have'
            ]

            for text in texts:
                sample = self.generate_next(text)
                print('%s... -> %s' % (text, sample))

        self.model = build_model()

        # get the training history after fitting the model
        history = self.model.fit(train_x, train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 validation_data=(val_x, val_y),
                                 callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

        self.plot_history(history)

        # # wrap the model in a scikit-learn classifier
        # sci_model = KerasClassifier(build_fn=build_model)
        #
        # # define the grid search parameters
        # batch_size = [10, 20, 40, 60, 80, 128]
        #
        # param_grid = dict(batch_size=batch_size)
        # grid = GridSearchCV(estimator=sci_model, param_grid=param_grid, n_jobs=-1)
        # grid_result = grid.fit(train_x, train_y)
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))


    def plot_history(self, history):
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()

    # TODO find a way to generate text without having to provide initial input
    def generate_text(self, char_limit, **parameters):
        return self.generate_next('This is a test', 24)


    def generate_next(self, text, num_generated=10):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.model.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)

    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def import_data(self, data_path):

        # read each line as a text
        with open(data_path) as file_:
            texts = file_.readlines()

        tokenizer = Tokenizer(num_words=self.num_words_to_keep)
        tokenizer.fit_on_texts(texts)
        # builds a list of tokenizer indices, e.g. each sentence is translated to its token indices
        sequences = tokenizer.texts_to_sequences(texts)
        # the mapping from those indices to the words in the tweets
        self.word_to_index = tokenizer.word_index
        # invert
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        print('Found %s unique tokens.' % len(self.word_to_index))

        # map the data into the new space
        print('\nPreparing the data for LSTM...')
        x = np.zeros([len(texts), self.max_sentence_len], dtype=np.int32)
        y = np.zeros([len(texts)], dtype=np.int32)

        # apply padding
        sequences = pad_sequences(sequences, maxlen=self.max_sentence_len)

        for i, sentence in enumerate(sequences):
            for t, word in enumerate(sentence[:-1]):
                # get the index of the word
                x[i, t] = word
            # predict the last word
            y[i] = sentence[-1]


        print('x shape:', x.shape)
        print('y shape:', y.shape)

        # split the data into a training set and a validation set
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        nb_validation_samples = int(self.validation_percentage * x.shape[0])

        x_train = x[:-nb_validation_samples]
        y_train = y[:-nb_validation_samples]
        x_val = x[-nb_validation_samples:]
        y_val = y[-nb_validation_samples:]

        return x_train, y_train, x_val, y_val

    def import_embeddings(self,embedding_file):
        embeddings_index = {}
        f = open(embedding_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        return embeddings_index

    def word2idx(self, word):
        return self.word_to_index[word]

    def idx2word(self, idx):
        return self.index_to_word[idx]

if __name__ == '__main__':
    test = GLove_LSTM()
    test.train('/home/klaus-michael/git_self/sad/data/clean_trump.csv')