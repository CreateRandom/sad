import random


def randomSentences(file_path, amount):
    sentences = []
    for x in range(amount):
        sentences.append(randomSentence(file_path))
    return sentences


def randomSentence(file_path):
        text = open(file_path)
        sentences = text.read().split("\n")
        sentence = random.choice(sentences)
        return sentence.lower()


if __name__ == '__main__':
    files = ['./data/clean_trump.csv',
             './data/clean_cnn.csv'
             ]
    amount = 7
    for filePath in files:
        print(filePath)
        sentences = randomSentences(filePath, amount)
        for sentence in sentences:
            print(sentence)
