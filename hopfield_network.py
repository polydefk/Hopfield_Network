import numpy as np
import Utils


class Hopfield(object):
    def __init__(self, memory_patterns):
        self.memory_patterns = memory_patterns
        self.weight_matrix = self.update_weights()

    def initialize_weights(self):
        """ Initilize the weights with zeros except the diagonal wich is NaN
            :returns the initialized matrix matrix
        """
        N = self.memory_patterns.shape[1]
        matrix = np.zeros((N, N))
        np.fill_diagonal(matrix, 0)
        return matrix

    def update_weights(self):

        """ updates the weight matrix through hebbian weights change
            :returns the matrix
        """

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = self.initialize_weights()

        w = self.weight_matrix
        N = self.memory_patterns.shape[1]

        for pattern in memory_patterns:
            temp = np.matrix.copy(w)
            for i in range(N):
                for j in range(N):
                    if i < j:
                        temp[i][j] = temp[j][i] = np.dot(pattern[i], pattern[j])

            w = np.add(w, temp)

        # self.weight_matrix = w
        # print(w)
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
        """:returns a boolean if the model is stable from matrix input x"""

        answers = np.full(self.memory_patterns.shape[0], False, dtype=bool)

        for i, pattern in enumerate(self.memory_patterns):
            answers[i] = self.recall(pattern)

        answer = np.all(answers)

        return answer

    def update(self, x):
        weights = self.weight_matrix
        random_matrix = np.random.randint(0, len(x), (1, len(x)))
        result = [0] * len(x)
        for i in range(len(random_matrix[0])):
            temp = np.dot(weights[random_matrix[0][i]], x)
            if temp >= 0:
                result[i] = 1
            else:
                result[i] = -1
        print(result)
        return result



if __name__ == '__main__':
    memory_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                                [-1, -1, -1, -1, -1, 1, -1, -1],
                                [-1, 1, 1, -1, -1, 1, -1, 1]])

    hopfield = Hopfield(memory_patterns)

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









