import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title):
    image = np.reshape(image, (32, 32))

    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.grid(True)
    # plt.savefig('pictures/3_2/{}.png'.format(title.replace(' ', '_')))
    plt.show()


def distort_data(pattern, percentage):
    dim = pattern.size

    n_dist = int(dim * percentage)

    idxs = np.random.choice(np.arange(dim), n_dist)

    dist_pattern = pattern.copy()

    dist_pattern[idxs] = pattern[idxs] * (-1)
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


def plot_energy_line(energy, legend_names, title):
    # fig config
    plt.figure()
    plt.grid(True)

    plt.plot(energy)

    plt.xlabel('Iterations')
    plt.ylabel('Energy')

    plt.title(title)
    plt.legend(legend_names, loc='upper left')

    plt.show()


if __name__ == "__main__":
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

    train_pattern = dataset[0].copy()

    distort_data(train_pattern, 0.1)
