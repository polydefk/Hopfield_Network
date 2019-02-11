import numpy as np
import Utils


class Hopfield(object):
    def __init__(self, memory_patterns):
        self.memory_patterns = memory_patterns
        self.weight_matrix = self.create_weight_matrix()

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal wich is NaN
            :returns the initialized matrix matrix
        """
        N = self.memory_patterns.shape[1]
        matrix = np.zeros((N, N))
        np.fill_diagonal(matrix, np.nan)
        return matrix

    def create_weight_matrix(self):

        """ Creates the weight matrix through hebbian weights change
            :returns the matrix
        """

        w = self.initialize_weights()
        N = self.memory_patterns.shape[1]

        for pattern in memory_patterns:
            temp = np.matrix.copy(w)
            for i in range(N):
                for j in range(N):
                    if i < j:
                        temp[i][j] = temp[j][i] = np.dot(pattern[i], pattern[j])

            w = np.add(w, temp)

        # self.weight_matrix = w

        return w

    def recall(self, x):
        """:returns a boolean if the model is stable from input x"""

        weights = self.weight_matrix
        values_to_check = np.zeros(weights.shape[0])

        for i, w in enumerate(weights):
            values_to_check[i] = Utils.activation_dynamic(w, x)

        answer = np.array_equal(x, values_to_check)
        return answer

    def recall_all(self):
        """:returns an array of booleans if the model is stable from input matrix x"""

        answers = np.full(self.memory_patterns.shape[0], False, dtype=bool)

        for i, pattern in enumerate(self.memory_patterns):
            answers[i] = self.recall(pattern)

        return answers


if __name__ == '__main__':
    memory_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                                [-1, -1, -1, -1, -1, 1, -1, -1],
                                [-1, 1, 1, -1, -1, 1, -1, 1]])

    hopfield = Hopfield(memory_patterns)

    for pattern in memory_patterns:
        hopfield.recall(pattern)

    hopfield.recall_all()
