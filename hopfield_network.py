import random
import numpy as np
<<<<<<< HEAD

=======
import Utils
import random as rand
>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042

class Hopfield(object):
    def __init__(self, memory_patterns, method='Batch'):
        self.memory_patterns = memory_patterns

        self.method = method

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal wich is NaN
            :returns the initialized matrix matrix
        """
        N = self.memory_patterns.shape[1]
        matrix = np.zeros((N, N))
<<<<<<< HEAD

=======
        np.fill_diagonal(matrix, 0)
>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042
        return matrix

    def train(self, memory_patterns=None):

        """ updates the weight matrix through hebbian weights change
            :returns the matrix
        """
        if memory_patterns is None:
            memory_patterns = self.memory_patterns

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = self.initialize_weights()

        if self.method is 'Batch':
            for pattern in memory_patterns:
                self.weight_matrix = np.add(np.outer(np.transpose(pattern), pattern), self.weight_matrix)
                np.fill_diagonal(self.weight_matrix, 0)

    def recall(self, pattern, i=-1, n_iterations=None):
        """:returns the attractor if this pattern is attractor
            or the pattern that minimizes the error
            or nothing"""

        error = 1
        iter = 0
        update = []

<<<<<<< HEAD
        if n_iterations is None:
            n_iterations = int(np.log(self.weight_matrix.shape[0])) + 1
=======
        # self.weight_matrix = w
        # print(w)
        return w
>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042

        if self.method is 'Batch':
            update = self.update_batch

        if self.method is "Random":
            update = self.update_random

        while error > 0:

            new_x = update(pattern)

            if i > 0:
                error = np.sum(np.abs(np.subtract(new_x, self.memory_patterns[i])))

            iter += 1

            if np.array_equal(pattern, new_x):
                return pattern

            pattern = new_x.copy()

            if iter == n_iterations:
                break

<<<<<<< HEAD
        if i > 0:
            return pattern
=======
    def update(self, x):
        weights = self.weight_matrix
        random_matrix = rand.sample(range(0, len(x)), len(x))
        # random_matrix = np.random.randint(0, len(x), (1, len(x)))
        result = [0] * len(x)
        for i in range(len(random_matrix)):
            temp = np.dot(weights[random_matrix[i]], x)
            if temp >= 0:
                result[i] = 1
            else:
                result[i] = -1
        print(result)
        return result


>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042

        # print("End of recall epochs needed {}".format(iter))

    def update_batch(self, x):
        """:returns"""
        result = np.dot(self.weight_matrix, x)
        result[result >= 0] = 1.
        result[result < 0] = -1.

        return result

    def update_random(self, x):
        rand_index = random.sample(range(0, len(x)), len(x))

<<<<<<< HEAD
        result = np.zeros(len(x))

        for i in range(len(rand_index)):
            temp = np.dot(self.weight_matrix[rand_index[i]], x)
            if temp >= 0:
                result[i] = 1
            else:
                result[i] = -1

        return result


if __name__ == '__main__':
    pass
=======
    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]

    input = hopfield.update(x1d)
    for i in range(100):
        input2 = hopfield.update(input)
        if np.array_equal(np.asarray(input), np.asarray(input2)):
            print(i)
            break
        else:
            input = input2









>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042
