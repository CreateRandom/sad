import gensim
import os

from gensim.scripts.glove2word2vec import glove2word2vec
import keras.backend as K
import numpy as np

def perplexity(y_true, y_pred):
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    pp = K.pow(np.e, cross_entropy)
    return pp

def convert_glove_to_w2v(glove_file):
    word2vec_output_file = glove_file + '.w2v'
    if(not os.path.isfile(word2vec_output_file)):
        # save as file
        glove2word2vec(glove_file, word2vec_output_file)

    # load the w2v file and return
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

def downsample(x,y, n_to_draw):
    idx = np.random.choice(np.arange(len(x)), n_to_draw, replace=False)
    return x[idx], y[idx]