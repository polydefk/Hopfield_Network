import itertools
import numpy as np
import Utils
from hopfield_network import Hopfield

np.random.seed(0)


def check_stability(hopfield, distorted, memory_patterns):
    if len(distorted.shape) < 2:
        distorted = np.reshape(distorted, (1, len(distorted)))

    stability = []
    for i, pattern in enumerate(distorted):
        value, _ = hopfield.recall(pattern)

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
        value, _ = hopfield.recall(pattern)
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

    hopfield = Hopfield(train_patterns)
    hopfield.train()
    stability = check_stability(hopfield, train_patterns, train_patterns)
    print(stability)

    if test_3_1:
        hopfield = Hopfield(train_patterns)
        hopfield.train()

        p10 = dataset[9].copy()
        recall_p10, _ = hopfield.recall(p10)

        Utils.display_image(p10, title='actual picture p10')
        Utils.display_image(recall_p10, title='recalled picture p10')

        p2 = dataset[1].copy()
        p3 = dataset[2].copy()
        p11 = dataset[10].copy()

        recall_p11, _ = hopfield.recall(p11)

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

        recalled, _ = hopfield.recall(p10, n_iterations=500)
        Utils.display_image(recalled, 'Asynchronous random update. closest to first training example')


# SOMETHING MAY BE WRONG NO SENSE MAKES THE ENERGY PLOTS
def ex_3_3__1_until_3():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns)
    hopfield.train()

    # 3.3.1
    print('energy at the di􏰂fferent attractors')
    print(hopfield.calculate_energy(train_patterns[0]))
    print(hopfield.calculate_energy(train_patterns[1]))
    print(hopfield.calculate_energy(train_patterns[2]))

    # 3.3.2
    print('energy at the points of the distorted patterns')
    print(hopfield.calculate_energy(dataset[3]))
    print(hopfield.calculate_energy(dataset[4]))
    print(hopfield.calculate_energy(dataset[5]))
    print(hopfield.calculate_energy(dataset[6]))
    print(hopfield.calculate_energy(dataset[7]))

    # 3.3.3 Follow how the energy changes from iteration to iteration
    # when you use the sequential update rule to approach an attractor.

    # _, energy = hopfield.recall(train_patterns[0], n_iterations=100, method='Async', calculate_energy=True)
    # Utils.plot_energy_line(energy, 'Energy', 'Energy using non-random asynchronous update to approach an attractor.')

    # _, energy = hopfield.recall(dataset[3], n_iterations=100, method='Async', calculate_energy=True)
    # Utils.plot_energy_line(energy, 'Energy', 'Energy non-random asynchronous update to approach a test point.')

    # _, energy = hopfield.recall(train_patterns[0], n_iterations=100, method='Random', calculate_energy=True)
    # Utils.plot_energy_line(energy, 'Energy', 'Energy using random asynchronous update to approach an attractor.')

    _, energy = hopfield.recall(dataset[3], n_iterations=100, method='Random', calculate_energy=True)
    Utils.plot_energy_line(energy, 'Energy', 'Energy random asynchronous update to approach a test point.')

    _, energy = hopfield.recall(dataset[5], n_iterations=100, method='Random', calculate_energy=True)
    Utils.plot_energy_line(energy, 'Energy', 'Energy random asynchronous update to approach a test point.')


def ex_3_3_4():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns, random_weights=True)
    hopfield.train()

    start_state = np.random.randn(len(dataset[0]))

    recalled, energy = hopfield.recall(start_state, n_iterations=100, method='Random', calculate_energy=True)
    Utils.plot_energy_line(energy, 'Energy', 'Energy random asynchronous update using random weights')


def ex_3_3_5():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns, make_weights_symmetric=True)
    hopfield.train()

    start_state = np.random.randn(len(dataset[0]))

    recalled, energy = hopfield.recall(start_state, n_iterations=100, method='Random', calculate_energy=True)
    Utils.plot_energy_line(energy, 'Energy', 'Energy random asynchronous update using symmetric weights')


if __name__ == "__main__":
    # ex_3_1_1()
    # ex_3_1_2()
    # ex_3_1_3()
    # ex_3_2()
    # ex_3_3__1_until_3()

    ex_3_3_4()
    ex_3_3_5()
