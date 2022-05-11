import pandas as pd


class Vector:
    def __init__(self, train_file, test_file):
        self.removed_chars = [',','.','!','?', 'I', ' are ', ' is ', ' were ', ' was ', '\r' ,'\n']
        self.train_file = train_file
        self.test_file = test_file

    def split_strings(self, x):
        if type(x) != str:
            return []
        for char in self.removed_chars:
            x = x.replace(char, " ")
        values = x.split(" ")
        return set([i for i in values if i])

    def get_words(self):
        train = pd.read_csv(self.train_file, sep='\t')
        test = pd.read_csv(self.test_file, sep='\t')
        data = pd.concat([train, test])

        data = data['benefitsReview'] + " " + data['sideEffectsReview'] + " " + data['commentsReview']
        print(data[80])
        data = data.apply(self.split_strings)

        words = []
        for row in data:
            words += list(row)

        words.sort()
        return set(words)