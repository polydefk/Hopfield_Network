import random
import numpy as np
import Utils


class Hopfield(object):
    def __init__(self, memory_patterns, method='Batch'):
        self.memory_patterns = memory_patterns

        self.method = method

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal which is NaN
            :returns the initialized matrix matrix
        """
        N = self.memory_patterns.shape[1]
        matrix = np.zeros((N, N))

        return matrix

    def train(self, memory_patterns=None):

        """ updates the weight matrix through hebbian weights change
            :returns the matrix
        """
        if memory_patterns is None:
            memory_patterns = self.memory_patterns

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = self.initialize_weights()

        for pattern in memory_patterns:
            self.weight_matrix = np.add(np.outer(np.transpose(pattern), pattern), self.weight_matrix)
        # np.fill_diagonal(self.weight_matrix, 0)

    def recall(self, pattern, i=-1, n_iterations=None):
        """:returns the attractor if this pattern is attractor
            or the pattern that minimizes the error
            or nothing"""

        # error = 1
        iter = 0
        update = []

        if n_iterations is None:
            n_iterations = int(np.log(self.weight_matrix.shape[0])) + 1

        if self.method is 'Batch':
            update = self.update_batch

        if self.method is "Random":
            update = self.update_random

        while True:

            new_x = update(pattern)

            # if i > 0:
            #     error = np.sum(np.abs(np.subtract(new_x, self.memory_patterns[i])))

            iter += 1
            print(iter)
            #
            # if np.array_equal(pattern, new_x):
            #     return pattern

            pattern = new_x.copy()

            if iter % 100 == 0:
                Utils.display_image(pattern, '')

            if iter == n_iterations:
                break

        return pattern

        # print("End of recall epochs needed {}".format(iter))

    def update_batch(self, x):
        """:returns"""
        result = np.dot(self.weight_matrix, x)
        result[result >= 0] = 1.
        result[result < 0] = -1.

        return result

    def update_random(self, x):
        rand_index = random.sample(range(0, len(x)), len(x))

        result = np.zeros(len(x))

        for i in range(len(x)):
            index = random.randint(0, len(x) - 1)
            temp = np.dot(self.weight_matrix[index], x)
            if temp >= 0:
                result[i] = 1
            else:
                result[i] = -1
        return result


if __name__ == '__main__':
    pass
