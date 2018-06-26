from Markov.markov_norder import Markov
import language_check

params = {'order': 3}
models = [Markov]
models = [m(**params) for m in models]
nTweets = 15
# set them all up
trainFiles = [#'./data/clean_trump.csv',
              './data/clean_cnn.csv'
              ]

for model in models:
    for filePath in trainFiles:
        print(filePath)
        mistakes = 0
        fileRange = model.train(filePath)
        for x in range(0, nTweets):
            fakeTweet = model.generate_text(280, filePath)
            print(fakeTweet)
            tool = language_check.LanguageTool('en-US')
            matches = tool.check(fakeTweet)
            mistakes += len(matches)
        avgMistakes = mistakes/nTweets
        # print(avgMistakes)

'''
# mistakes made in original files
        lines = 0
        for line in open(filePath):
            lines += 1
            tool = language_check.LanguageTool('en-US')
            matches = tool.check(line.lower())
            mistakes += len(matches)
        avgMistakes = mistakes/lines
        print(lines)
        print(mistakes)'''
