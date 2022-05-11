import numpy as np
import pandas as pd


class Vector:
    def __init__(self, train_file, test_file):
        self.removed_chars = [',','.','!','?', 'I', ' are ', ' is ', ' were ', ' was ', '\r' ,'\n', 'the', 'and']
        self.train_file = train_file
        self.test_file = test_file
        self.__init_words()

    def split_strings(self, x):
        if type(x) != str:
            return []
        for char in self.removed_chars:
            x = x.replace(char, " ")
        values = x.split(" ")
        return [i for i in values if i]

    def __init_words(self):
        train = pd.read_csv(self.train_file, sep='\t')
        test = pd.read_csv(self.test_file, sep='\t')
        data = pd.concat([train, test])

        self.train_rating = train["rating"]
        self.test_rating = test["rating"]

        self.train_rating = self.train_rating.tolist()
        self.test_rating = self.test_rating.tolist()

        train = train['commentsReview'] + " " + train['sideEffectsReview'] + " " + train['commentsReview']
        test = test['commentsReview'] + " " + test['sideEffectsReview'] + " " + test['commentsReview']

        train = train.apply(self.split_strings)
        test = test.apply(self.split_strings)
        data = data.apply(self.split_strings)


        train_words = []
        for row in train:
            train_words.append(row)

        test_words = []
        for row in test:
            test_words.append(row)

        self.all_words = set()
        words = train_words + test_words
        for i in words:
            for j in i:
                self.all_words.add(j)

        self.train_words = []
        for row in train_words:
            temp = dict.fromkeys(self.all_words, 0)
            for word in row:
                temp[word] += 1
            values = temp.values()
            self.train_words.append(list(values))

        self.test_words = []
        for row in test_words:
            temp = dict.fromkeys(self.all_words, 0)
            for word in row:
                temp[word] += 1
            values = temp.values()
            self.test_words.append(list(values))


