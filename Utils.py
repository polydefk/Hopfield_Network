import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title):
    image = np.reshape(image, (32, 32))
    plt.clf()
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.grid(True)
    plt.savefig('pictures/3_2_3/{}.png'.format(title.replace(' ', '_')))
    # plt.show()


def distort_data(pattern, percentage):
    dim = pattern.size

    n_dist = int(dim * percentage)
    numbers = np.arange(dim)
    np.random.shuffle(numbers)

    # idxs = np.random.choice(np.arange(dim), n_dist)

    dist_pattern = pattern.copy()

    dist_pattern[:int(numbers * percentage)] *= (-1)
    # np.random.shuffle(dist_pattern[:n_dist])

    return dist_pattern


def activation_dynamic(w, x, theta=0):
    value = 0
    for j in range(w.shape[0]):
        if ~np.isnan(w[j]):
            value += np.dot(w[j], x[j]) - theta

    value = np.sign(value)
    value[value == 0] = 1

    return value


def check_performance(original, prediction):
    N = len(original)

    # accuracy = np.count_nonzero(original == prediction)

    accuracy = 0

    for i in range(N):
        if original[i] == prediction[i]:
            accuracy += 1

    accuracy = int(accuracy * 100 / N)

    return accuracy


def plot_energy_line(energy, legend_names, title):
    # fig config
    plt.figure()
    plt.grid(True)
    plt.clf()
    plt.plot(energy)

    plt.xlabel('Iterations')
    plt.ylabel('Energy')

    plt.title(title)
    plt.legend(legend_names, loc='upper left')

    plt.show()


def generate_random_pattern(N=1024):
    pattern = np.random.normal(0, 1, size=N)
    pattern[pattern >= 0] = 1
    pattern[pattern < 0] = -1
    # pattern = np.array(np.where(np.random.normal(0, 1, size=N) >= 0, 1, -1))
    return pattern


def plot_accuracy(accuracy, title):
    plt.figure()
    plt.grid(True)

    plt.plot(accuracy)

    plt.xlabel('Number of patterns')
    plt.ylabel('Accuracy')
    plt.ylim([-1, 101])
    plt.title(title)
    # plt.legend('', loc='upper left')

    plt.show()


if __name__ == "__main__":
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    #
    train_pattern = dataset[0].copy()
    #
    # dist = distort_data(train_pattern, 0.1)
    # check_performance(train_pattern, dist)
    pattern = generate_random_pattern(len(train_pattern))
    display_image(pattern,'')