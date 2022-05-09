# Student Name: Samet Umut Yigitoglu
# Student ID: 260201056

import numpy as np
import pandas as pd
import random

train_data_path = "./data/drugLibTrain_raw.tsv"
test_data_path = "./data/drugLibTest_raw.tsv"

# HYPERPARAMETERS
input_size = 12
output_size = 10
hidden_layer_sizes = [input_size, input_size]
learning_rate = 0.5
number_of_epochs = 5

# 'He Initialization' for weights and initial 0.01 values for all biases
W_B = {
    'W1': np.random.randn(hidden_layer_sizes[0], input_size) * np.sqrt(2/hidden_layer_sizes[0]),
    'b1': np.ones((hidden_layer_sizes[0], 1)) * 0.01,
    'W2': np.random.randn(hidden_layer_sizes[1], hidden_layer_sizes[0]) * np.sqrt(2/hidden_layer_sizes[0]),
    'b2': np.ones((hidden_layer_sizes[1], 1)) * 0.01,
    'W3': np.random.randn(output_size, hidden_layer_sizes[1]) * np.sqrt(2/hidden_layer_sizes[1]),
    'b3': np.ones((output_size, 1)) * 0.01
}

inputs = [np.random.uniform(-1, 1) for _ in range(input_size)]

h = [[np.random.uniform(-1, 1) for _ in range(input_size)],
     [np.random.uniform(-1, 1) for _ in range(hidden_layer_sizes[0])],
     [np.random.uniform(-1, 1) for _ in range(hidden_layer_sizes[1])],
     [np.random.uniform(-1, 1) for _ in range(output_size)]]


# initialize weights
w = [[[np.random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_layer_sizes[0])],
     [[np.random.uniform(-1, 1) for _ in range(hidden_layer_sizes[0])] for _ in range(hidden_layer_sizes[1])],
     [[np.random.uniform(-1, 1) for _ in range(hidden_layer_sizes[1])] for _ in range(output_size)]]


# sigmoid function
def activation_function(layer):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return sigmoid(layer)


def derivation_of_activation_function(signal):
    sigmoid_derivation = lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
    # activation = [sigmoid_derivation(np.dot(h[signal - 1], w[signal - 1][i])) for i in range(len(h[signal]))]
    return sigmoid_derivation(signal)

def loss_function(true_labels, probabilities):
    pass

# RSS 1/2 * sigma((target - output)**2)
def rss(layer):
    pass

# sum-of-squares error (rss) is used to turn activations into probability distribution

def derivation_of_loss_function(true_labels, probabilities):
    pass

# the derivation should be with respect to the output neurons

def forward_pass(data):
    # h[0] = data
    for i in range(1, len(h) - 1):
        h[i] = [activation_function(np.dot(h[i - 1], w[i - 1][j])) for j in range(len(h[i]))]
    outputs = [np.dot(h[len(h) - 2], w[len(h) - 2][j]) for j in range(output_size)]
    sorted_outputs = sorted(outputs)
    max_index = outputs.index(sorted_outputs[0])
    h[len(h) - 1] = [0] * output_size
    h[len(h) - 1][max_index] = 1
    print(h[len(h) - 1])




# [hidden_layers] is not an argument, but it is up to you how many hidden layers to implement.
# so replace it with your desired hidden layers
def backward_pass(input_layer, hidden_layers, output_layer, loss):
    pass

def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
        index = 0

        # Same thing about [hidden_layers] mentioned above is valid here also
        for data, labels in zip(train_data, train_labels):
            predictions, [hidden_layers] = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, predictions)
            backward_pass(data, [hidden_layers], predictions, loss_signals)
            loss = loss_function(labels, predictions)

            if index % 2000 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1

    # return losses


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        prediction, _, _ = forward_pass(data)
        predictions.append(prediction)
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

forward_pass(h[0])

if __name__ == "__main__":

    train_data = pd.read_csv(train_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')
    train_x = train_data["commentsReview"]
    train_y = train_data['rating']
    test_x = ["commentsReview"]
    print(train_x)
    test_y = test_data['rating']

    # creating one-hot vector notation of labels. (Labels are given numeric in the dataset)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y) - 1):
        new_train_y[i][train_y[i]] = 1

    for i in range(len(test_y) - 1):
        new_test_y[i][test_y[i]] = 1

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
