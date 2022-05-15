# Student Name: Samet Umut Yigitoglu
# Student ID: 260201056

import numpy as np
import pandas as pd


class Vector:
    """
    This class takes train and test file paths. Reads the files and tokenizes the words. And then creates a set for every
    word in the comments and reviews. And creates a list for train and test data.
    """

    def __init__(self, train_file, test_file):
        # These words and characters will be removed from the list.
        self.removed_words = [',', '.', '!', '?', 'I', ' are ', ' is ', ' were ', ' was ', '\r', '\n', 'the', 'and']
        self.train_file = train_file
        self.test_file = test_file
        self.__init_words()

    # Method for tokenizing
    def split_strings(self, x):
        if type(x) != str:
            return []
        for char in self.removed_words:
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

        train = train['commentsReview'] + " " + train['sideEffectsReview'] + " " + train['benefitsReview']
        test = test['commentsReview'] + " " + test['sideEffectsReview'] + " " + test['benefitsReview']

        # Tokenize words
        train = train.apply(self.split_strings)
        test = test.apply(self.split_strings)
        data = data.apply(self.split_strings)

        train_words = []
        for row in train:
            train_words.append(row)

        test_words = []
        for row in test:
            test_words.append(row)

        # Create a set for words. This set will be the template of the dictionary that I will use as an input layer.
        self.all_words = set()
        words = train_words + test_words
        for i in words:
            for j in i:
                self.all_words.add(j)

        # List for training words
        self.train_words = []
        for row in train_words:
            temp = dict.fromkeys(self.all_words, 0)
            for word in row:
                temp[word] += 1
            values = temp.values()
            self.train_words.append(list(values))

        # List for test words
        self.test_words = []
        for row in test_words:
            temp = dict.fromkeys(self.all_words, 0)
            for word in row:
                temp[word] += 1
            values = temp.values()
            self.test_words.append(list(values))


train_data_path = "./data/drugLibTrain_raw.tsv"
test_data_path = "./data/drugLibTest_raw.tsv"

# Create a vector class to read files
vector = Vector(train_data_path, test_data_path)
# Take lists from the vector object
all_words = vector.all_words
train_x = vector.train_words
train_y = vector.train_rating
test_x = vector.test_words
test_y = vector.test_rating

# HYPERPARAMETERS
input_size = len(all_words)
output_size = 10
hidden_layer_sizes = [200, 150]
learning_rate = 0.05
number_of_epochs = 50

# Initialize weights and biases
W_B = {
    'W1': np.random.randn(hidden_layer_sizes[0], input_size) * np.sqrt(2 / hidden_layer_sizes[0]),
    'b1': np.ones((hidden_layer_sizes[0], 1)) * 0.01,
    'W2': np.random.randn(hidden_layer_sizes[1], hidden_layer_sizes[0]) * np.sqrt(2 / hidden_layer_sizes[0]),
    'b2': np.ones((hidden_layer_sizes[1], 1)) * 0.01,
    'W3': np.random.randn(output_size, hidden_layer_sizes[1]) * np.sqrt(2 / hidden_layer_sizes[1]),
    'b3': np.ones((output_size, 1)) * 0.01
}


# sigmoid function
def activation_function(layer):
    return 1 / (1 + np.exp(-layer))


def derivation_of_activation_function(signal):
    return (1 - signal) * signal


def loss_function(true_labels, probabilities):
    value = true_labels - probabilities["Y"]
    return rss(value)


def rss(layer):
    return np.sum(layer ** 2)


def derivation_of_loss_function(true_labels, probabilities):
    return probabilities["Y"] - true_labels


def forward_pass(data):
    # Calculate first layer
    z1 = np.dot(data, W_B['W1'].T) + W_B['b1'].T
    a1 = activation_function(z1)

    # Calculate second layer
    z2 = np.dot(a1, W_B['W2'].T) + W_B['b2'].T
    a2 = activation_function(z2)

    # Calculate prediction
    y = np.dot(a2, W_B['W3'].T) + W_B['b3'].T

    forward_results = {"Z1": z1,
                       "A1": a1,
                       "Z2": z2,
                       "A2": a2,
                       "Y": y}

    return forward_results


def backward_pass(input_layer, output_layer, loss):
    # Calculate deltas
    output_delta = loss
    z3_delta = np.dot(output_delta, W_B['W3'])
    a2_delta = z3_delta * derivation_of_activation_function(output_layer['A2'])
    z2_delta = np.dot(a2_delta, W_B['W2'])
    a1_delta = z2_delta * derivation_of_activation_function(output_layer['A1'])

    # Adjust weights and biases
    W_B['W3'] -= learning_rate * np.outer(output_layer['A2'], output_delta).T
    W_B['b3'] -= learning_rate * np.sum(output_delta, axis=1, keepdims=True)
    W_B['W2'] -= learning_rate * np.outer(output_layer['A1'], a2_delta).T
    W_B['b2'] -= learning_rate * np.sum(a2_delta, axis=1, keepdims=True)
    W_B['W1'] -= learning_rate * np.outer(input_layer, a1_delta).T
    W_B['b1'] -= learning_rate * np.sum(a1_delta, axis=1)


def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
        index = 0
        for data, labels in zip(train_data, train_labels):
            output = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, output)
            backward_pass(data, output, loss_signals)

            if index % 400 == 0:  # at each 400th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        prediction = forward_pass(data)
        predictions.append(prediction["Y"])
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction))

    # Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
    # And turn predictions into one-hot encoded

    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(true_labels[i]):  # if 1 is in same index with ground truth
            true_pred += 1

    return true_pred / len(predictions)


if __name__ == "__main__":
    # creating one-hot vector notation of labels. (Labels are given numeric in the dataset)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))
    for i in range(len(train_y) - 1):
        new_train_y[i][train_y[i] - 1] = 1

    for i in range(len(test_y) - 1):
        new_test_y[i][test_y[i] - 1] = 1
    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%75-%25)
    valid_x = np.asarray(train_x[int(0.75 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.75 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.75 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.75 * len(train_y))])

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y))
