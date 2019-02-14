import itertools
import numpy as np
import Utils
from hopfield_network import Hopfield


def check_stability(hopfield, distorted, memory_patterns):
    if len(distorted.shape) < 2:
        distorted = np.reshape(distorted, (1, len(distorted)))

    stability = []
    for i, pattern in enumerate(distorted):
        value = hopfield.recall(pattern)

        # print("The case of x{}d is {}"
        #       .format(i, np.array_equal(value, memory_patterns[i])))

        stability.append(np.array_equal(value, memory_patterns[i]))

    return stability


def ex_3_1_1():
    memory_patterns = np.array([[-1., -1., 1., -1., 1., -1., -1., 1.],
                                [-1., -1., -1., -1., -1., 1., -1., -1.],
                                [-1., 1., 1., -1., -1., 1., -1., 1.]])

    hopfield = Hopfield(memory_patterns)
    hopfield.train()

    distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                          [1, 1, -1, -1, -1, 1, -1, -1],
                          [1, 1, 1, -1, 1, 1, -1, 1]])

    print(check_stability(hopfield, distorted, memory_patterns))



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

    uniques = np.unique(np.array(attractors), axis=0)

    print("The number of attractors is {}".format(len(uniques)))


def ex_3_1_3():
    memory_patterns = np.array([[-1., -1., 1., -1., 1., -1., -1., 1.],
                                [-1., -1., -1., -1., -1., 1., -1., -1.],
                                [-1., 1., 1., -1., -1., 1., -1., 1.]])

    hopfield = Hopfield(memory_patterns)
    hopfield.train()

    distorted = np.array([[1, 1, 1, 1, 1, -1, 1, 1]])

    distorted = np.repeat(distorted, 3, axis=0)

    print(check_stability(hopfield, distorted, memory_patterns))


def ex_3_2():

    test_3_1 = False
    test_3_2 = True
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    # hopfield = Hopfield(train_patterns)
    # hopfield.train()
    # stability = check_stability(hopfield, train_patterns, train_patterns)
    # print(stability)

    if test_3_1:
        hopfield = Hopfield(train_patterns)
        hopfield.train()

        p10 = dataset[9].copy()
        recall_p10 = hopfield.recall(p10)

        Utils.display_image(p10, title='actual picture p10')
        Utils.display_image(recall_p10, title='recalled picture p10')

        p2 = dataset[1].copy()
        p3 = dataset[2].copy()
        p11 = dataset[10].copy()

        recall_p11 = hopfield.recall(p11)

        Utils.display_image(p2, title='actual picture p2')
        Utils.display_image(p3, title='actual picture p3')
        Utils.display_image(p11, title='actual picture p11 (Mixture p2 and p3)')
        Utils.display_image(recall_p11, title='recalled picture p11')

        print(np.all(check_stability(hopfield, recall_p10, train_patterns)))
        print(np.all(check_stability(hopfield, recall_p11, train_patterns)))

    if test_3_2:
        hopfield = Hopfield(train_patterns, method='Random')
        hopfield.train()
        p10 = dataset[9].copy()

        Utils.display_image(p10, title='actual picture p10')

        hopfield.recall(p10, n_iterations=5000)

def ex_3_3_1():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns)
    hopfield.train()

    print(hopfield.calculate_energy(train_patterns[0]))
    print(hopfield.calculate_energy(train_patterns[1]))
    print(hopfield.calculate_energy(train_patterns[2]))



if __name__ == "__main__":
    # ex_3_1_1()
    # ex_3_1_2()
    # ex_3_1_3()
    # ex_3_2()
    ex_3_3_1()
