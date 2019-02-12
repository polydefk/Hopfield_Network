import itertools
import numpy as np
from hopfield_network import Hopfield


def ex_3_1_1():
    memory_patterns = np.array([[-1., -1., 1., -1., 1., -1., -1., 1.],
                                [-1., -1., -1., -1., -1., 1., -1., -1.],
                                [-1., 1., 1., -1., -1., 1., -1., 1.]])

    hopfield = Hopfield(memory_patterns)
    hopfield.train()

    distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                          [1, 1, -1, -1, -1, 1, -1, -1],
                          [1, 1, 1, -1, 1, 1, -1, 1]])

    for i, pattern in enumerate(distorted):
        value = hopfield.recall(pattern, i)
        print("The case of x{}d is {}".format(i, np.array_equal(value, memory_patterns[i])))


def ex_3_1_2():
    memory_patterns = np.array([[-1., -1., 1., -1., 1., -1., -1., 1.],
                                [-1., -1., -1., -1., -1., 1., -1., -1.],
                                [-1., 1., 1., -1., -1., 1., -1., 1.]])

    hopfield = Hopfield(memory_patterns)
    hopfield.train()

    lst = [list(i) for i in itertools.product([-1., 1.], repeat=8)]
    lst = np.array(lst)

    attractors = []
    for i in range(len(lst)):
        pattern = lst[i]
        value = hopfield.recall(pattern)
        if value is not None:
            attractors.append(value)

    attractors = np.array(attractors)
    uniques = np.unique(attractors, axis=0)

    print("The number of attractors is {}".format(len(uniques)))





if __name__ == "__main__":
    ex_3_1_1()
    ex_3_1_2()
