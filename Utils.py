import numpy as np
import matplotlib.pyplot as plt
import pylab as pb


def display_image(image, title):
    image = np.reshape(image, (32, 32))
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.grid(True)
    # plt.savefig('pictures/3_2_3/{}.png'.format(title.replace(' ', '_')))
    plt.show()


def distort_data(pattern, percentage):
    N = pattern.size
    n_dist = int(N * percentage)
    idxs = np.random.choice(np.arange(N), n_dist)
    dist_pattern = pattern.copy()
    dist_pattern[idxs] *= -1

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
    pattern = pattern

    pattern[pattern >= 0] = 1
    pattern[pattern < 0] = -1
    # pattern = np.array(np.where(np.random.normal(0, 1, size=N) >= 0, 1, -1))
    return pattern


def plot_accuracy(accuracy, title):
    plt.figure()
    plt.grid(True)

    # plt.plot(np.arange(0,300,10),accuracy)
    plt.plot(np.arange(0,len(accuracy),1),accuracy)

    plt.xlabel('Number of patterns')
    plt.ylabel('Accuracy')
    plt.ylim([-1, 101])
    plt.title(title)
    # plt.legend('', loc='upper left')

    plt.show()

def plot_stable_patterns(patterns, title):
    plt.figure()
    plt.grid(True)

    plt.plot(patterns)

    plt.xlabel('Number of patterns')
    plt.ylabel('Number of stable patterns')
    plt.ylim([0, 1])
    plt.title(title)
    # plt.legend('', loc='upper left')

    plt.show()



def show_images(images, rows, save_dir, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))

    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images / float(rows)), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.pause(0.00001)
    plt.savefig(save_dir)


def plot_multiple_accuracy(accuracy, pattern_number, legend_names, title):
    plt.figure()
    plt.grid(True)

    patterns = np.arange(0, pattern_number, 1)

    for i in range(len(accuracy)):
        plt.plot(patterns, accuracy[i], alpha=0.5)

    plt.xlabel('Number of patterns')
    plt.ylabel('Accuracy')
    plt.ylim([-1, 101])
    plt.title(title)
    plt.legend(legend_names, loc='upper right')
    # plt.legend('', loc='upper left')

    plt.show()


if __name__ == "__main__":
    dataset = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    #
    train_pattern = dataset[0].copy()
    #
    dist = distort_data(train_pattern, 0.9
                        )
    images = [np.reshape(train_pattern, (32, 32)), np.reshape(dist, (32, 32))]
    show_images(images, rows=1)
