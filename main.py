from Markov.markov_norder import Markov
import language_check

params = {'order': 3}
models = [Markov]
models = [m(**params) for m in models]
# set them all up

for model in models:
    trumpRange = model.train('./data/clean_trump.csv')
    for x in range(0, 10):
        fakeTweet = model.generate_text(char_limit=280)
        # TODO test generated text: amount of spelling + grammar mistakes
        # To compare results and optimize parameters
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(fakeTweet)
        mistakes = len(matches)
        print(fakeTweet)
        print(mistakes)
