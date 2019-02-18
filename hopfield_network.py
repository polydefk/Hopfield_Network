import random
import numpy as np
import Utils
import matplotlib.pyplot as plt

class Hopfield(object):
    def __init__(self, memory_patterns, activity=None, method=None, random_weights=False, make_weights_symmetric=False, sparse_weight=False):
        self.memory_patterns = memory_patterns
        self.method = method
        self.random_weights = random_weights
        self.make_weights_symmetric = make_weights_symmetric
        self.sparse_weight = sparse_weight
        self.activity = activity

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal which is NaN
            :returns the initialized matrix matrix
        """
        N = self.memory_patterns.shape[1]
        matrix = np.zeros((N, N))

        if self.random_weights:
            matrix = np.random.normal(0, 1, size=matrix.shape)
            # np.fill_diagonal(matrix, 0)

        if self.make_weights_symmetric:
            matrix = np.random.normal(0, 1, size=matrix.shape)
            matrix = 0.5 * (matrix + np.transpose(matrix))
            # np.fill_diagonal(matrix, 0)

        return matrix

    def train(self, memory_patterns=None):

        """ updates the weight matrix through hebbian weights change
            :returns the matrix
        """
        if memory_patterns is None:
            memory_patterns = self.memory_patterns

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = self.initialize_weights()

        if self.sparse_weight:
            for pattern in memory_patterns:
                pattern = np.subtract(pattern,self.activity)
                self.weight_matrix = np.add(np.outer(pattern.T, pattern), self.weight_matrix)
        else:
            for pattern in memory_patterns:
                self.weight_matrix = np.add(np.outer(np.transpose(pattern), pattern), self.weight_matrix)

        np.fill_diagonal(self.weight_matrix, 0)

    def recall(self, pattern, n_iterations=None, method=None, calculate_energy=False, plot=False):
        """:returns the attractor if this pattern is attractor
            or the pattern that minimizes the error
            or nothing"""

        iter = 1
        update = []

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = self.initialize_weights()

        if n_iterations is None:
            n_iterations = int(np.log(self.weight_matrix.shape[0])) + 1

        energy = []

        if method is not None:
            self.method = method

        if self.method is 'Batch':
            update = self.update_synchronous_batch

        if self.method is "Random":
            update = self.update_random

        if self.method is "Async":
            update = self.update_async

        if self.sparse_weight:
            update = self.update_sparse

        while True:
            new_x = []
            if self.sparse_weight:
                new_x = update(pattern, 0)
            else:
                new_x = update(pattern)

            if calculate_energy:
                energy.append(self.calculate_energy(new_x))


            if pattern.tolist() in self.memory_patterns.tolist():
                print("recalled")
                plt.clf()
                plt.imshow(new_x.reshape((32, 32)).T)
                plt.pause(1)
                return [pattern, np.array(energy)]

            if np.array_equal(pattern, new_x):
                print("Converged but not recalled")
                return [pattern, np.array(energy)]


            pattern = new_x.copy()

            if iter == n_iterations:
                break

            iter += 1
        return [pattern, np.array(energy)]

    def update_synchronous_batch(self, x):
        """:returns"""
        result = np.dot(self.weight_matrix, x)
        result[result >= 0] = 1.
        result[result < 0] = -1.

        return result

    def update_random(self, pattern):
        dimension = pattern.shape[0]
        new = pattern.copy()
        for count in range(3000):
            i = np.random.randint(0, dimension)
            new[i] = np.where(np.dot(self.weight_matrix[i], pattern) > 0, 1, -1)

            if i % 100 == 0:
                plt.clf()
                plt.imshow(new.reshape((32, 32)).T)
                plt.pause(1e-4)

            pattern = new.copy()

        return new

    def update_async(self, pattern):
        dimension = pattern.shape[0]

        new = pattern.copy()
        for i in range(dimension):
            new[i]= np.where(np.dot(self.weight_matrix[i], pattern) > 0, 1, -1)
            if i % 100 == 0:
                plt.clf()
                plt.imshow(new.reshape((32, 32)).T)
                plt.pause(1e-4)

            pattern = new.copy()

        return new


    def calculate_energy(self, training_data):

        result = np.matmul(self.weight_matrix, training_data)
        return -np.dot(result, training_data)

    def update_sparse(self, x, theta):
        value = 0
        for j in range(self.weight_matrix.shape[0]):
            value += np.dot(self.weight_matrix[j], x[j]) - theta

        value = np.sign(value)
        value[value == 0] = 1
        value = 0.5 + 0.5 * value

        return value


if __name__ == '__main__':
    pass
