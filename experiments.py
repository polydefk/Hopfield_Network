import itertools
import numpy as np
import Utils
from hopfield_network import Hopfield
import matplotlib.pyplot as plt

np.random.seed(0)


def check_stability(hopfield, distorted, memory_patterns):
    if len(distorted.shape) < 2:
        distorted = np.reshape(distorted, (1, len(distorted)))

    stability = []
    for i, pattern in enumerate(distorted):
        value, _ = hopfield.recall(pattern, 0)

        print("The case of x{}d is {}"
              .format(i, np.array_equal(value, memory_patterns[i])))

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

    distorted = np.array([[-1, -1, 1, 1, -1, 1, 1, 1]])

    # distorted = np.repeat(distorted, 3, axis=0)
    _, recalled = check_stability(hopfield, distorted, memory_patterns)

    print(recalled)


def ex_3_2():
    test_3_1 = False
    test_3_2 = True
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()
    #
    # hopfield = Hopfield(train_patterns)
    # hopfield.train()
    # stability = check_stability(hopfield, train_patterns, train_patterns)

    if test_3_1:
        hopfield = Hopfield(train_patterns)
        hopfield.train()

        p10 = dataset[9].copy()
        recall_p10, _ = hopfield.recall(p10)

        Utils.display_image(p10, title='actual picture p10')
        Utils.display_image(recall_p10, title='recalled picture p10')

        p2 = dataset[1].copy()
        p1 = dataset[0].copy()
        p3 = dataset[2].copy()
        p11 = dataset[10].copy()

        recall_p11, _ = hopfield.recall(p11)

        Utils.display_image(p2, title='actual picture p2')
        Utils.display_image(p1, title='actual picture p1')
        Utils.display_image(p3, title='actual picture p3')
        Utils.display_image(p11, title='actual picture p11 (Mixture p2 and p3)')
        Utils.display_image(recall_p11, title='recalled picture p11')

        # print(np.all(check_stability(hopfield, recall_p10, train_patterns)))
        # print(np.all(check_stability(hopfield, recall_p11, train_patterns)))

    if test_3_2:
        hopfield = Hopfield(train_patterns, method='Async')
        hopfield.train()
        p10 = dataset[9].copy()

        # Utils.display_image(p10, title='actual picture p10')

        recalled, _ = hopfield.recall(p10, n_iterations=500, plot=True)
        Utils.display_image(recalled, 'Asynchronous random update. closest to first training example')


# SOMETHING MAY BE WRONG NO SENSE MAKES THE ENERGY PLOTS
def ex_3_3__1_until_3():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns)
    hopfield.train()
    # 3.3.3 Follow how the energy changes from iteration to iteration
    # when you use the sequential update rule to approach an attractor.
    for i in range(len(dataset)):
        recalled, energy = (
            hopfield.recall(dataset[i], number_of_dataset=i, n_iterations=10, method='Random',
                            calculate_energy=True, plot=True))
        print("Energy for pattern {} is {}".format(i + 1, energy))

        Utils.plot_energy_line(np.array(energy), 'Energy',
                               'Energy random asynchronous update to approach a test point.')


def ex_3_3_4():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()
    dim = 1024
    start_state = np.random.randn(dim)
    method = ['Random', 'Async']
    for m in method:
        hopfield = Hopfield(start_state, random_weights=True, method=m, make_weights_symmetric=True)
        hopfield.train()
        # hopfield.weight_matrix = np.random.randn(dim, dim)

        pattern = Utils.generate_random_pattern(train_patterns.shape[1])
        recalled, energy = hopfield.recall(pattern, 1, calculate_energy=True, n_iterations=50)
        # print(energy)
        # Utils.display_image(recalled, '')
        # recalled, energy = hopfield.recall(dataset[10], n_iterations=100, method='Random', calculate_energy=True)
        Utils.plot_energy_line(energy, 'Energy', 'Energy using {} update using symmetric weights'.format(m))


def ex_3_3_5():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns, make_weights_symmetric=True)

    start_state = np.random.normal(0, 1, len(dataset[0]))

    recalled, energy = hopfield.recall(dataset[10], n_iterations=100, method='Random', calculate_energy=True)
    Utils.plot_energy_line(energy, 'Energy', 'Energy random asynchronous update using symmetric weights')


def ex_3_4():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    train_patterns = dataset[0:3].copy()

    hopfield = Hopfield(train_patterns)
    hopfield.train()

    for idx, pattern in enumerate(train_patterns):
        for percentage in range(100):
            percentage = percentage / 100 + 0.01

            dist_pic = Utils.distort_data(pattern, percentage)

            recalled, energy = hopfield.recall(dist_pic,  number_of_dataset=1, method='Batch')

            images = [np.reshape(dist_pic, (32, 32)), np.reshape(recalled, (32, 32))]
            titles = ["Distorted with {}% noise".format(percentage), "Recalled picture"]

            fname = 'pattern_{0}_percent_{1}'.format(idx + 1, round(percentage, 3))
            save_dir = 'pictures/3_4/{0}.png'.format(fname)

            Utils.show_images(images, save_dir=save_dir, rows=1, titles=titles)


def plot_image():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    pattern = dataset[0].copy()
    plt.imshow(pattern.reshape((32, 32)).T)
    plt.show()


def ex_3_5():
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    test_1 = False
    test_2 = True

    if test_1:
        accuracy = []

        for i in range(3, 8):
            train_patterns = dataset[0:i].copy()

            hopfield = Hopfield(train_patterns)
            hopfield.train()
            local_accuracy = []
            for j, pattern in enumerate(train_patterns):
                # dist_pattern = Utils.distort_data(pattern, 0)

                value, _ = hopfield.recall(pattern, n_iterations=15)
                local_accuracy.append(Utils.check_performance(pattern, value))

            print(
                "Accuracy {0} for {1} training patterns ".format(round(np.mean(local_accuracy), 1), i))

            accuracy.append(np.mean(local_accuracy))
        Utils.plot_accuracy(np.array(accuracy), '')
    if test_2:

        train_patterns = dataset[0:3].copy()
        randoms = []
        for pt in train_patterns:
            randoms.append(pt)
        values = []

        for i in range(1024):

            randoms.append(np.array(Utils.generate_random_pattern()))

            local_accuracy = []

            hopfield = Hopfield(np.array(randoms))
            hopfield.train()
            for j, pattern in enumerate(randoms):
                # dist_pattern = Utils.distort_data(pattern, 0)

                value, _ = hopfield.recall(pattern, n_iterations=15)
                local_accuracy.append(Utils.check_performance(pattern, value))
            value = np.mean(np.array(local_accuracy))
            values.append(value)

        Utils.plot_accuracy(np.array(values), '{OLA}')


def ex_3_6():
    memory_patterns = np.array([[0., 0., 1., 0., 1., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 1., 0., 0.],
                                [0., 1., 1., 0., 0., 1., 0., 1.]])

    hopfield = Hopfield(memory_patterns, 0.1, sparse_weight=True)
    hopfield.train()
    for pattern in memory_patterns:
        value, _ = hopfield.recall(pattern, 0)
        print(value)
    # distorted = np.array([[1, 1, 1, 1, 1, 0, 1, 1]])
    #
    # distorted = np.repeat(distorted, 3, axis=0)


if __name__ == "__main__":
    # ex_3_1_1()
    # ex_3_1_2()
    # ex_3_1_3()
    # ex_3_2()
    # ex_3_3__1_until_3()
    # ex_3_3_4()
    # ex_3_3_5()
    ex_3_4()
    # ex_3_5()
    # run_3_1_seq()
    # ex_3_6()
