# (c) Ernesto Elsäßer 2019

import csv
import numpy as np

class WeightClassifier:

    def __init__(self, hidden_neurons = 6, epsilon = 0.1, learning_rate = 0.3):
        self.net = MultilayerPerceptron(6, hidden_neurons, 2, epsilon, learning_rate)
        self.classes = ["Untergewicht", "Normalgewicht", "Uebergewicht"]
    
    def load_data(self, filename):
        self.xs = np.zeros((0,6))
        self.targets = np.zeros((0,2))
        self.raw = []

        print("reading CSV file ...")
        with open(filename, newline='') as file:
            data = csv.reader(file, delimiter=';')
            next(data) # skip first line
            for row in data:
                gender = 1 if row[0] == 'w' else 0
                height = self.normalize(int(row[1]), 140, 200)
                age = self.normalize(int(row[2]), 18, 100)
                weight = self.normalize(int(row[3]), 20, 150)
                strength_sports = 1 if row[4] == "Kraftsport" else 0
                endurance_sports = 1 if row[4] == "Ausdauersport" else 0
                underweight = 1 if row[5] == self.classes[0] else 0
                overweight = 1 if row[5] == self.classes[2] else 0
                self.xs = np.append(self.xs, [[gender, height, age, weight, strength_sports, endurance_sports]], 0)
                self.targets = np.append(self.targets, [[underweight, overweight]], 0)
                self.raw.append(" | ".join(row))

        print("loaded " + str(len(self.raw)) + " samples from " + filename)
        
    def normalize(self, value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

    def train(self, from_index = 0, to_index = 10000):
        train_xs = self.xs[from_index:to_index]
        train_targets = self.targets[from_index:to_index]
        print("training on " + str(train_xs.shape[0]) + " samples ...")
        before = np.datetime64('now')
        self.net.train(train_xs, train_targets, max_iterations = 15)
        after = np.datetime64('now')
        print("finished in " + str(after - before))

    def test(self, from_index = 0, to_index = 10000, verbose = False):
        train_xs = self.xs[from_index:to_index]
        train_targets = self.targets[from_index:to_index]
        count = train_xs.shape[0]
        correct_count = 0

        for i in range(count):
            expected_class = self.classify(self.targets[i])
            (_, y) = self.net.propagate(self.xs[i])
            predicted_class = self.classify(y)
            correct = expected_class == predicted_class
            if correct:
                correct_count += 1
            if verbose:
                print(str(i+1) + ": " + self.raw[i] +  (" --- " if correct else " -X- ") 
                      + self.classes[predicted_class] + " " + str(y))

        accuracy = correct_count / count
        print("test results: {0}/{1} correct ({2:.0%})".format(correct_count, count, accuracy))

    def classify(self, y):
            if y[0] > 0.5:
                return 0
            if y[1] > 0.5:
                return 2
            return 1

class MultilayerPerceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, epsilon, learning_rate):
        self.INPUT = 0
        self.HIDDEN = 1
        self.OUTPUT = 2
        self.output_dim = output_dim
        weights_from_in = 2 * np.random.random((input_dim + 1, hidden_dim)) - 1
        weights_from_hidden = 2 * np.random.random((hidden_dim + 1, output_dim)) - 1
        self.weights_from = [weights_from_in, weights_from_hidden]
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    @staticmethod
    def transfer(x):
        return 1/(1 + np.exp(-x)) # sigmoid

    @staticmethod
    def transfer_derived(x):
        return x * (1-x) # sigmoid dervied

    def propagate_layer(self, layer, levels):
        levels_ext = np.append(levels, 1) # treshold
        w = self.weights_from[layer]
        return self.transfer(np.dot(levels_ext, w))

    def propagate(self, x):
        h = self.propagate_layer(self.INPUT, x)
        y = self.propagate_layer(self.HIDDEN, h)
        return (h, y)

    def energy(self, y, target):
        return np.sum(np.square(target - y)) / 2

    def adjust_weights(self, layer, levels_left, levels_right, error):
        scaled_error = error * self.transfer_derived(levels_right)
        delta = self.learning_rate * np.outer(levels_left, scaled_error)
        delta_ext = np.append(delta, np.zeros((1, error.shape[0])), 0)
        self.weights_from[layer] += delta_ext
        return scaled_error # reuse for next layer

    def backpropagate(self, x, h, y, target):
        error = target - y
        scaled_error = self.adjust_weights(self.HIDDEN, h, y, error)
        weights_from_hidden = self.weights_from[self.HIDDEN]
        hidden_error = scaled_error.dot(weights_from_hidden[:-1].T)
        self.adjust_weights(self.INPUT, x, h, hidden_error)

    def train(self, xs, targets, max_iterations):
        count = xs.shape[0]
        iteration = 0
        pending = 1
        while pending > 0 and iteration < max_iterations:
            iteration += 1
            pending = 0
            total_error = 0
            
            for i in range(count):
                x = xs[i]
                target = targets[i]
                (h, y) = self.propagate(x)
                energy = self.energy(y, target)
                if energy > self.epsilon:
                    self.backpropagate(x, h, y, target)
                    (_, y) = self.propagate(x)
                    energy = self.energy(y, target)
                    if (energy > self.epsilon):
                        pending += 1
                total_error += energy

            convergence = (count - pending)/count
            avg_error = total_error/count
            print("round {0} - convergence: {1:.0%} - mean error: {2}".format(iteration, convergence, avg_error))