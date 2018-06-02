from Pipeline.Model import Model

# https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b
__author__ = 'maxim'

import numpy as np
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

class Word2Vec_LSTM(Model):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.model = Sequential()
        self.word_model = None
        self.n_epochs = parameters.get('n_epochs',20)
        self.batch_size = parameters.get('batch_size',128)
        self.embedding_size = parameters.get('embedding_size',100)
        self.embedding_path = parameters.get('embedding_path', None) #"/home/klaus-michael/Downloads/w2v.twitter.27B.50d.txt")
        self.max_sentence_len = parameters.get('max_sentence_len',40)
    def train(self, data_path, **parameters):
        print('\nPreparing the sentences...')

        with open(data_path) as file_:
            docs = file_.readlines()
        print('Num tweets:', len(docs))

        # split the tweets into sentences of at most 40 words
        # transformed_docs = [doc.lower().translate(string.punctuation).split() for doc in docs]
        # compute the number of words in the longest sentence
        # max_sentence_len = len(max(transformed_docs,key=len))
        sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:self.max_sentence_len]] for doc in
                     docs]

        print('Num sentences:', len(sentences))

        # if no pre-trained embedding was provided, train one on our training data

        if(self.embedding_path is None):
            print('\nTraining word2vec...')
            self.word_model = gensim.models.Word2Vec(sentences, size=self.embedding_size, min_count=1, window=5, iter=100)

        # NB: this works, but is agonisingly slow due to the high size of the embedding most pre-trained models have
        else:
            print('\nLoading word2vec...')
            self.word_model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_path, binary=False)

        pretrained_weights = self.word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape
        print('Result embedding shape:', pretrained_weights.shape)

        # print('Checking similar words:')
        # for word in ['I', 'clinton', 'Trump']:
        #  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
        #  print('  %s -> %s' % (word, most_sim128ilar))

        print('\nPreparing the data for LSTM...')
        train_x = np.zeros([len(sentences), self.max_sentence_len], dtype=np.int32)
        train_y = np.zeros([len(sentences)], dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence[:-1]):
                train_x[i, t] = self.word2idx(word)
            train_y[i] = self.word2idx(sentence[-1])
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)

        print('\nTraining LSTM...')
        self.model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        self.model.add(LSTM(units=emdedding_size))
        self.model.add(Dense(units=vocab_size))
        self.model.add(Activation('softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # print a neat little summary
        print(self.model.summary())

        def on_epoch_end(epoch, _):
            print('\nGenerating text after epoch: %d' % epoch)
            texts = [
                'a',
            ]
            for text in texts:
                sample = self.generate_next(text)
                print('%s... -> %s' % (text, sample))

        self.model.fit(train_x, train_y,
                  batch_size=self.batch_size,
                  epochs=self.n_epochs,
                  callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

    def word2idx(self, word):
        if not (word in self.word_model.wv.vocab):
            # TODO think about what to do here instead
            return self.word_model.wv.vocab['blank'].index
        return self.word_model.wv.vocab[word].index

    def idx2word(self, idx):
        return self.word_model.wv.index2word[idx]

    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_next(self, text, num_generated=10):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.model.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)

    # TODO make sure char limit is adhered to
    def generate_text(self, char_limit, **parameters):
        return self.generate_next(text='This is a test',num_generated= 24)