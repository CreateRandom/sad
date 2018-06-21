import os
import sys
from itertools import chain, repeat, islice

from keras import Sequential, optimizers
from keras.callbacks import LambdaCallback
# a sci-kit learn wrapper for Keras classifier
from keras.wrappers.scikit_learn import KerasClassifier
from losswise.libs import LosswiseKerasCallback
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

from Pipeline.Model import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional, Dropout, regularizers
from LSTM.W2VUtils import perplexity, convert_glove_to_w2v
import numpy as np
import losswise

# compare https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# and https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b

# This class reads in word embeddings and uses them as the input space
# for a word-level prediction task. Two modes are currently supported:
# 1) Readout mode: Predictions based on final readout layer, weights trained
# 2) Dense mode: Predictions based on LSTM output, converted back to to word based on closest embedding
# Use the do_readout parameter to control which mode is used

class Embedding_LSTM(Model):
    # our losswise API key, enables us to do online loss tracing
    losswise.set_api_key('EFW1YNI7H')

    def __init__(self, **parameters):
        super().__init__(**parameters)
        # frequency cutoff, if set to None --> keep all words
        self.num_words_to_keep = parameters.get('num_words_to_keep',None)
        # max length of sentence to use, first n - 1 tokens --> input, last token --> output
        self.max_sentence_len = parameters.get('max_sentence_len',20)
        # percentage of inputs to use for validation
        self.validation_percentage = parameters.get('validation_perc',0.05)

        # learning rate
        self.learning_rate = parameters.get('learning_rate', 0.1)

        self.n_epochs = parameters.get('n_epochs', 10)
        self.batch_size = parameters.get('batch_size', 64)

        # path of the pre-trained glove embeddings
        if not 'glove_embedding_path' in parameters:
            raise Exception('For running the gLove model, please provide the path of the embedding file'
                            'as a parameter.')
        self.embedding_path = parameters.get('glove_embedding_path', None)

        # full word model
        self.word_model = convert_glove_to_w2v(self.embedding_path)
        self.embedding_size = self.word_model.vector_size

        self.word_to_index = None


        # whether to include a readout layer or not
        self.do_readout = parameters.get('do_readout',False)

    def train(self, data_path, **parameters):

        # train samples and validation set
        train_x, train_y, val_x, val_y = self.import_data(data_path)
        # Generate custom embedding matrix
        # idea: only use words that are in the training data --> save memory
        embeddings_index = self.import_embeddings(self.embedding_path)
        vocab_size = len(self.word_to_index) + 1

        # generate the embedding matrix
        embedding_matrix = np.zeros((vocab_size, self.embedding_size))
        for word, i in self.word_to_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load it into a layer for mapping the data
        embedding_layer = Embedding(len(self.word_to_index) + 1,
                                    self.embedding_size,
                                    weights=[embedding_matrix],
                                    trainable=False)

        # if we don't have a readout layer: map the target values into vector space
        if(not self.do_readout):
            train_y = self.map_indices_to_embedding(train_y,embedding_matrix)
            val_y = self.map_indices_to_embedding(val_y,embedding_matrix)

        print('\nTraining LSTM...')

        def build_readout_model(learning_rate):
            model = Sequential()
            # word embeddings
            model.add(embedding_layer)
            # LSTM
            model.add(LSTM(units=self.embedding_size))

            # readout layer with softmax
            # TODO think about adding regularization
            model.add(Dense(units=vocab_size)) #, kernel_regularizer=regularizers.l2(0.01)))
            model.add(Activation('softmax'))

            # momentum: shown to be irrelevant
            adam = optimizers.adam(lr=learning_rate, decay=1e-6)
            # use xent between predicted class and actual class
            model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=[perplexity])
            return model

        def build_dense_model(learning_rate):
            model = Sequential()
            # word embeddings
            model.add(embedding_layer)
            # LSTM
            model.add(LSTM(units=self.embedding_size))
            # TODO think about adding dropout
            # model.add(Dropout(0.2))

            # use RMSPROP, better for regression?
            sgd = optimizers.rmsprop(lr=learning_rate, decay=1e-6)
            # TODO see how we can add perplexity here as well
            model.compile(optimizer=sgd, loss='mean_squared_error')
            return model

        # callback function for when an epoch ends
        def on_epoch_end(epoch, _):
            print('\nGenerating text after epoch: %d' % epoch)
            texts = [
                "the move could", 'it was stated', 'the media have'
            ]

            for text in texts:
                sample = self.generate_next(text)
                print('%s... -> %s' % (text, sample))

        if(self.do_readout):
            self.model = build_readout_model(learning_rate=self.learning_rate)
        else:
            self.model = build_dense_model(learning_rate=self.learning_rate)

        # get the training history after fitting the model
        history = self.model.fit(train_x, train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 validation_data=(val_x, val_y),
                                 callbacks=[LambdaCallback(on_epoch_end=on_epoch_end), LosswiseKerasCallback()])

        self.plot_history(history)

        # experiments with GridSearch, didn't work, look into it again
        # link: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

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

    # helper for plotting model history, also see Losswise
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
        return self.generate_next('unk', 24)

    # generate next predicted word based on some input text
    def generate_next(self, text, num_generated=10):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.model.predict(x=np.array(word_idxs))
            if(self.do_readout):
                idx = self.sample_from_softmax_layer(prediction[-1], temperature=0.1)
            else:
                idx = self.sample_from_embedding(prediction[-1], temperature=0.1)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)

    # sample from softmax output based on temperature
    # higher temperature --> more randomness
    def sample_from_softmax_layer(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # TODO add temperature
    def sample_from_embedding(self,pred_vector, temperature = 0.0):
        selected = 'unknown'
        # sample five words, for each check whether in original corpus
        values = self.word_model.most_similar(positive=[pred_vector], topn=5)
        for word, sim in values:
            if word in self.word_to_index:
                selected = word
                break
        return self.word2idx(selected)

    def map_indices_to_embedding(self, indices, embedding):
        toReturn = np.zeros((len(indices),self.embedding_size))
        count = 0
        for y_index in np.nditer(indices):
            toReturn[count] = embedding[y_index]
            count = count + 1
        return toReturn

    # read in data from data path
    def import_data(self, data_path):

        # read each line as a text
        with open(data_path) as file_:
            texts = file_.readlines()

        # use Keras Tokenizer to index words, filter out special characters
        tokenizer = Tokenizer(num_words=self.num_words_to_keep,oov_token='<unk>', filters='‘—’”…“!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
        tokenizer.fit_on_texts(texts)
        # builds a list of tokenizer indices, e.g. each sentence is translated to its token indices
        sequences = tokenizer.texts_to_sequences(texts)
        # the mapping from those indices to the words in the tweets
        self.word_to_index = tokenizer.word_index
        # invert
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        print('Found %s unique tokens.' % len(self.word_to_index))

        # experimented with chunking input, inconclusive so far

        # chunked_sequences = []
        #
        # for sequence in sequences:
        #     sequence_chunks = list(chunks(sequence,self.max_sentence_len))
        #     chunked_sequences.extend(sequence_chunks)

        # apply padding
        sequences = pad_sequences(sequences, maxlen=self.max_sentence_len)

        # map the data into the new space
        print('\nPreparing the data for LSTM...')
        x = np.zeros([len(sequences), self.max_sentence_len], dtype=np.int32)
        y = np.zeros([len(sequences)], dtype=np.int32)

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

    # helper for reading in embeddings
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

# helper method for chunkizing sequences
def chunks(chunkable, n):
    for i in range(0, len(chunkable), n):
        yield chunkable[i:i+n]