from LSTM.Embedding_LSTM import Embedding_LSTM

from Markov.markov_norder import Markov

# To use word embeddings, download them from here: https://nlp.stanford.edu/projects/glove/
# and unzip the desired one into the folder
params = {'order': 3,
          'glove_embedding_path': '../pre_trained/glove.6B.200d.txt',
          'n_epochs': 40}
models = [Embedding_LSTM, Markov]
# set them all up

models = [m(**params) for m in models]

for model in models:
    trumpRange = model.train('./data/clean_trump.csv')
    fakeTweet = model.generate_text(char_limit=280)

    #
    # for x in range(0, 10):
    #     fakeTweet = model.generate_text(char_limit=280)
    #     # TODO test generated text: amount of spelling + grammar mistakes
    #     # To compare results and optimize parameters
    #     tool = language_check.LanguageTool('en-US')
    #     matches = tool.check(fakeTweet)
    #     mistakes = len(matches)
    #     print(fakeTweet)
    #     print(mistakes)
