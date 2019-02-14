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


def activation_dynamic(w, x, theta=0):
    value = 0
    for j in range(w.shape[0]):
        if ~np.isnan(w[j]):
            value += np.dot(w[j], x[j]) - theta

    value = np.sign(value)
    value[value == 0] = 1

    return value
