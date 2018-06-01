from Markov.markov_norder import Markov

params = {'order' : 3}
models = [Markov]
# set them all up
models = [m(**params) for m in models]

for model in models:
    # TODO include absolute path here
    model.train('/home/klaus-michael/git_self/sad/pres-speech/obama')
    print(model.generate_text(char_limit=140))