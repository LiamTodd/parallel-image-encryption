import math
import os
import random
import sys
import threading
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image

POOL_SIZE = 4
SECRET_KEY_BITS = 480
NUM_SUB_KEYS = 12
NUM_LATTICES = 10
THREADS = 4
DEFAULT_IMG = "./img/landscape_1.png"


def generate_secret_key(length):
    key = ""
    for _ in range(length):
        key += str(random.randint(0, 1))
    return key


def logistic_map(x, mu):
    return mu * x * (1 - x)


def generate_coupled_map_logistic_lattices(K, m, n, par):
    # secret key divided into subkeys
    k = [K[i : i + SECRET_KEY_BITS // NUM_SUB_KEYS] for i in range(NUM_SUB_KEYS)]

    # get mu and e
    temp_sum_mu = 0
    temp_sum_e = 0
    for i in range(SECRET_KEY_BITS // NUM_SUB_KEYS):
        temp_sum_mu += int(k[0][i]) * 2**i
        temp_sum_e += int(k[1][i]) * 2**i
    # for mu approaching 4 and e approaching 0, CML is optimal
    mu = 3.99 + (0.01 * temp_sum_mu / (2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS)))
    e = 0.1 * temp_sum_e / (2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS))

    x_0 = [0] * NUM_LATTICES
    for i in range(NUM_LATTICES):
        for j in range(SECRET_KEY_BITS // NUM_SUB_KEYS):
            x_0[i] += int(k[i + 2][j]) * 2**j
        x_0[i] /= 2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS)

    x = [[0] * (201 + max(n, n, par)) for _ in range(NUM_LATTICES)]
    for i in range(NUM_LATTICES):
        x[i][0] = x_0[i]

    # iterate 201+max(m,n,par) times and find xi in every lattice
    for i in range(1, 201 + max(m, n, par)):
        for lattice in range(NUM_LATTICES):
            if lattice == 0:
                x[lattice][i] = ((1 - e) * logistic_map(x[lattice][i - 1], mu)) + (
                    (e / 2)
                    * (
                        logistic_map(x[NUM_LATTICES - 1][i - 1], mu)
                        + logistic_map(x[lattice + 1][i - 1], mu)
                    )
                )
            elif lattice == NUM_LATTICES - 1:
                x[lattice][i] = ((1 - e) * logistic_map(x[lattice][i - 1], mu)) + (
                    (e / 2)
                    * (
                        logistic_map(x[lattice - 1][i - 1], mu)
                        + logistic_map(x[0][i - 1], mu)
                    )
                )
            else:
                x[lattice][i] = ((1 - e) * logistic_map(x[lattice][i - 1], mu)) + (
                    (e / 2)
                    * (
                        logistic_map(x[lattice - 1][i - 1], mu)
                        + logistic_map(x[lattice + 1][i - 1], mu)
                    )
                )

    return x


def permutate_data(img_data, M, N, S, H):
    sort_column = img_data[:, np.argsort(N)]
    cirshift_d = sort_column
    for i in range(len(N)):
        cirshift_d[:, i] = np.roll(cirshift_d[:, i], S[i])
    sort_rows = cirshift_d[np.argsort(M), :]
    cirshift_r = sort_rows
    for i in range(len(M)):
        cirshift_r[i, :] = np.roll(cirshift_r[i, :], H[i])
    return cirshift_r


def process_cmls(cmls, m, n, par):
    M = cmls[0][-m:]
    M = [math.floor(M[i] * pow(10, 10)) for i in range(len(M))]
    N = cmls[1][-n:]
    N = [math.floor(N[i] * pow(10, 10)) for i in range(len(N))]
    H = [M[i] % m for i in range(len(M))]
    S = [N[i] % n for i in range(len(N))]
    t = 5
    b = 3
    if (m > pow(10, t - 1) and m <= pow(10, t)) and (n > 0 and n <= pow(10, b)):
        M = [M[i] * pow(10, t) for i in range(len(M))]
        N = [N[i] * pow(10, b) for i in range(len(N))]
    M = [(M[i] + i) for i in range(m)]
    N = [(N[i] + i) for i in range(n)]
    A = cmls[2][-par:]
    B = cmls[3][-par:]
    D = cmls[4][-par:]
    E = cmls[5][-par:]
    return M, N, H, S, A, B, D, E


def TIDBD_diffuse(permutated_img_data, p, A, B, D, E):
    pass


def parse_args():
    show_images = False
    file_path = DEFAULT_IMG
    if len(sys.argv) > 1:
        show_images = sys.argv[1] == "1"
        if len(sys.argv) > 2:
            file_path = f"./img/{sys.argv[2]}"
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return 1
    return show_images, file_path


def main():
    show_images, file_path = parse_args()

    # convert image to greyscale
    image = Image.open(file_path).convert("L")
    img_data = np.asarray(image)
    par = 2 * THREADS + 4
    K = generate_secret_key(SECRET_KEY_BITS)
    m = img_data.shape[0]
    n = img_data.shape[1]
    cmls = generate_coupled_map_logistic_lattices(K, m, n, THREADS)
    M, N, H, S, A, B, D, E = process_cmls(cmls, m, n, par)
    permutated_img_data = permutate_data(img_data, M, N, S, H)
    diffused_img_data = TIDBD_diffuse(permutated_img_data, A, B, D, E)

    if show_images:
        image.show()
        permuted_image = Image.fromarray(permutated_img_data)
        permuted_image.show()
        diffused_img_data.show()

    return 0


if __name__ == "__main__":
    main()
