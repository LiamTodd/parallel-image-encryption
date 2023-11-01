import math

from src.constants import NUM_LATTICES, NUM_SUB_KEYS, SECRET_KEY_BITS
from src.utils import logistic_map


def process_cmls(cmls, m, n, par):
    """
    This function extracts the lattices from the CML, which will be used to seed the permutation and diffusion processes
    """
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


def generate_coupled_map_logistic_lattices(K, m, n, par):
    """
    This function generates the CML given a secret key, dimensions, and the level of parallelism
    """
    # secret key divided into subkeys
    k = [K[i : i + SECRET_KEY_BITS // NUM_SUB_KEYS] for i in range(NUM_SUB_KEYS)]

    # initialise lattice values using the subkeys
    x_0 = [0] * NUM_LATTICES
    for i in range(NUM_LATTICES):
        for j in range(SECRET_KEY_BITS // NUM_SUB_KEYS):
            x_0[i] += int(k[i + 2][j]) * 2**j
        x_0[i] /= 2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS)

    x = [[0] * (201 + max(n, n, par)) for _ in range(NUM_LATTICES)]
    for i in range(NUM_LATTICES):
        x[i][0] = x_0[i]

    # get mu and e
    temp_sum_mu = 0
    temp_sum_e = 0
    for i in range(SECRET_KEY_BITS // NUM_SUB_KEYS):
        temp_sum_mu += int(k[0][i]) * 2**i
        temp_sum_e += int(k[1][i]) * 2**i
    # for mu approaching 4 and e approaching 0, CML is optimal
    mu = 3.99 + (0.01 * temp_sum_mu / (2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS)))
    e = 0.1 * temp_sum_e / (2 ** (SECRET_KEY_BITS // NUM_SUB_KEYS))

    # apply chaos to lattices using logistic map
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
