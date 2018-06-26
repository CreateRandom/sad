class Model(object):

    def __init__(self, **parameters):

        pass

    def train(self, data_path, **parameters):

        pass

    def save(self, path):

        pass

    def load(self,model_file):

        pass

    # generates a text of at most max_char chars
    # for twitter, current max_char is 280
    def generate_text(self, char_limit, file_path, **parameters):

        pass
