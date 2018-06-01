from Markov.markov_norder import Markov
import language_check

params = {'order': 3}
models = [Markov]
# set them all up

for model in models:
    model.train('./data/obama')
    print(model.generate_text(char_limit=280))

    # TODO test generated text: amount of spelling + grammar mistakes
    # To compare results and optimize parameters
    # Example
    tool = language_check.LanguageTool('en-US')
    text = u'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
    matches = tool.check(text)
    mistakes = len(matches)
