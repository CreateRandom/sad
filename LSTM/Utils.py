import gensim
import os

from gensim.scripts.glove2word2vec import glove2word2vec
import keras.backend as K
import numpy as np
from matplotlib import pyplot


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

# helper method for chunkizing sequences
def chunks(chunkable, n):
    for i in range(0, len(chunkable), n):
        yield chunkable[i:i+n]

def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z

# helper for plotting model history, also see Losswise
def plot_history(history, name):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('../logs/' + name + '/' + 'loss.png')
    pyplot.close()