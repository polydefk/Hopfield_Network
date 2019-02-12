import itertools
import random

import numpy as np


class Hopfield(object):
    def __init__(self, memory_patterns, batch=True):
        self.memory_patterns = memory_patterns
        self.batch = batch
        # self.update_weights()

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal wich is NaN
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

        if self.batch:
            for pattern in memory_patterns:
                self.weight_matrix = np.add(np.outer(np.transpose(pattern), pattern), self.weight_matrix)
                np.fill_diagonal(self.weight_matrix, 0)

    def recall(self, x, i=-1, n_iterations=None):
        """:returns the prediction of how many values are wrong and the number of iterations
            it used to converge."""
        error = 1
        iter = 0
        update = []
        if n_iterations is None:
            n_iterations = int(np.log(self.weight_matrix.shape[0])) + 1

        if self.batch:
            update = self.update_batch

        while error > 0:

            new_x = update(x)
            # print("New x value  {}".format(new_x))
            # print("Real pattern {}".format(self.memory_patterns[i]))
            # print()
            if i > 0:
                error = np.sum(np.abs(np.subtract(new_x, self.memory_patterns[i])))
            # print(error)
            x = new_x.copy()
            iter += 1

            if iter is n_iterations:
                break


        print("End of recall epochs needed {}".format(iter))
        # print()
        return x

    def update_batch(self, x):

        result = np.dot(self.weight_matrix, x)
        # print(result)
        result[result >= 0] = 1.
        result[result < 0] = -1.

        return result

    def update_random(self, x):
        rand_index = random.sample(range(0, len(x)), len(x))

        result = np.zeros(len(x))

        for i in range(len(rand_index)):
            temp = np.dot(self.weight_matrix[rand_index[i]], x)
            if temp >= 0:
                result[i] = 1
            else:
                result[i] = -1

        return result


if __name__ == '__main__':
    memory_patterns = np.array([[-1., -1., 1., -1., 1., -1., -1., 1.],
                                [-1., -1., -1., -1., -1., 1., -1., -1.],
                                [-1., 1., 1., -1., -1., 1., -1., 1.]])

    hopfield = Hopfield(memory_patterns)
    hopfield.train()

    distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                          [1, 1, -1, -1, -1, 1, -1, -1],
                          [1, 1, 1, -1, 1, 1, -1, 1]])

    value = np.zeros(distorted.shape[1])
    for i, pattern in enumerate(distorted):
        value = hopfield.recall(pattern, i)
        print(np.array_equal(value, memory_patterns[i]))

    lst = [list(i) for i in itertools.product([-1., 1.], repeat=8)]
    lst = np.array(lst)



    attractors = []
    for i in range(len(lst)):

        pattern = lst[i]

        value = hopfield.recall(pattern)

        if not value.tolist() in attractors:
            attractors.append(value.tolist())

    print(len(attractors))

