import math
from multiprocessing.pool import ThreadPool

import numpy as np

from src.constants import POOL_SIZE
from src.utils import logistic_map


def diffuse(Fr, Ar, Br, Dr, Er):
    Gr = []
    Gr.append(Ar * pow(10, 5) % 256)
    Gr.append(Br * pow(10, 5) % 256)
    mu = 3.99 + 0.01 * Dr
    xi = Er
    for i in range(1, len(Fr) + 1):
        xi = logistic_map(xi, mu)
        Gr.append(
            (
                math.floor(Gr[i - 1] + pow(10, 5) * Gr[i] / 255 + xi * pow(10, 5))
                + Fr[i - 1]
            )
            % 256
        )
    Gr = Gr[2 : len(Fr) + 2]
    return Gr


def thread_func_1(Fr, A, B, D, E, r):
    return diffuse(Fr[r], A[r], B[r], D[r], E[r])


def thread_func_2(Gr_list, A, B, D, E, r, p):
    return diffuse(Gr_list[r - p - 4], A[r], B[r], D[r], E[r])


def TIDBD_diffuse(permutated_img_data, p, A, B, D, E, m, n):
    permutated_img_data = permutated_img_data.reshape(1, m * n)[0]

    # segment P1 into p groups with length t
    Fr = []
    if (m * n) % p == 0:
        t = m * n // p

    else:
        t = math.floor(m * n / p) + 1
        T = np.zeros((t * p) - (m * n), np.uint8)
        permutated_img_data = np.append(permutated_img_data, T)

    for r in range(p):
        Fr.append(permutated_img_data[r * t : (r + 1) * t])

    # first independent diffusion of each group in parallel
    pool = ThreadPool(POOL_SIZE)
    Gr_list = []
    for r in range(p):
        x = pool.apply_async(thread_func_1, (Fr, A, B, D, E, r))
        Gr_list.append(x.get())
    pool.close()
    pool.join()

    # exchange the first and last two pixels within each group
    for r in range(p):
        Gr_len = len(Gr_list[r])
        Gr_list[r][0], Gr_list[r][Gr_len - 1] = Gr_list[r][Gr_len - 1], Gr_list[r][0]
        Gr_list[r][1], Gr_list[r][Gr_len - 2] = Gr_list[r][Gr_len - 2], Gr_list[r][1]

    # bidirectional diffusion of the exchanged pixels
    GA1 = []
    GA2 = []

    for r in range(p):
        GA1.append(Gr_list[r][0])
        GA2.append(Gr_list[r][1])

    GB1 = diffuse(GA1, A[p + 1], B[p + 1], D[p + 1], E[p + 1])
    GB2 = diffuse(GA2, A[p + 3], B[p + 3], D[p + 3], E[p + 3])

    GC1 = np.flip(GB1)
    GC2 = np.flip(GB2)

    GD1 = diffuse(GC1, A[p + 2], B[p + 2], D[p + 2], E[p + 2])
    GD2 = diffuse(GC2, A[p + 4], B[p + 4], D[p + 4], E[p + 4])

    for r in range(p):
        Gr_list[r][0] = GD1[r]
        Gr_list[r][1] = GD2[r]

    # second independent diffusion of each group in parallel
    pool = ThreadPool(POOL_SIZE)
    O = []
    for r in range(p + 4, 2 * p + 4):
        x = pool.apply_async(thread_func_2, (Gr_list, A, B, D, E, r, p))
        O.append(x.get())
    pool.close()
    pool.join()

    # combine groups into diffused image C
    C = []
    for i in range(p):
        C.append(O[i])
    C = np.array(C)
    if (m * n) % p == 0:
        C = C.reshape(n, m)
    else:
        C = C[0 : m * n].reshape(m, n)

    return C, GD1, GD2


def decrypt_diffuse(Gr, Ar, Br, Dr, Er):
    Fr = []
    Gr_len = len(Gr)
    Gr = np.concatenate(([(Ar * pow(10, 5)) % 256], Gr))
    Gr = np.concatenate(([(Br * pow(10, 5)) % 256], Gr))
    mu = 3.99 + 0.01 * Dr
    xi = Er
    for i in range(1, Gr_len + 1):
        xi = logistic_map(xi, mu)
        Fr.append(
            (
                Gr[i + 1]
                - math.floor(Gr[i - 1] + pow(10, 5) * Gr[i] / 255 + xi * pow(10, 5))
            )
            % 256
        )
    return Fr


def thread_func_3(O, A, B, D, E, r, p):
    return decrypt_diffuse(O[r - p - 4], A[r], B[r], D[r], E[r])


def thread_func_4(Gr_list, A, B, D, E, r):
    return decrypt_diffuse(Gr_list[r], A[r], B[r], D[r], E[r])


def decrypt_diffusion(diffused_img_data, m, n, p, A, B, D, E, GD1, GD2):
    C = diffused_img_data.reshape(1, m * n)[0]
    O = []
    if (m * n) % p == 0:
        t = m * n // p
    else:
        t = math.floor(m * n / p) + 1
        T = np.zeros((t * p) - (m * n), np.uint8)
        C = np.append(C, T)

    for r in range(p):
        O.append(C[r * t : (r + 1) * t])

    # parallel decryption of second diffusion of groups
    pool = ThreadPool(POOL_SIZE)
    Gr_list = []
    for r in range(p + 4, 2 * p + 4):
        x = pool.apply_async(thread_func_3, (O, A, B, D, E, r, p))
        Gr_list.append(x.get())
    pool.close()
    pool.join()

    for r in range(p):
        GD1[r] = Gr_list[r][0]
        GD2[r] = Gr_list[r][1]
    GC1 = decrypt_diffuse(GD1, A[p + 2], B[p + 2], D[p + 2], E[p + 2])
    GC2 = decrypt_diffuse(GD2, A[p + 4], B[p + 4], D[p + 4], E[p + 4])

    GB1 = np.flip(GC1)
    GB2 = np.flip(GC2)

    GA1 = decrypt_diffuse(GB1, A[p + 1], B[p + 1], D[p + 1], E[p + 1])
    GA2 = decrypt_diffuse(GB2, A[p + 3], B[p + 3], D[p + 3], E[p + 3])

    for r in range(p):
        Gr_list[r][0] = GA1[r]
        Gr_list[r][1] = GA2[r]

    for r in range(p):
        Gr_len = len(Gr_list[r])
        Gr_list[r][0], Gr_list[r][Gr_len - 1] = Gr_list[r][Gr_len - 1], Gr_list[r][0]
        Gr_list[r][1], Gr_list[r][Gr_len - 2] = Gr_list[r][Gr_len - 2], Gr_list[r][1]

    # parallel decryption of first diffusion of groups
    pool = ThreadPool(POOL_SIZE)
    Fr = []
    for r in range(p):
        x = pool.apply_async(thread_func_4, (Gr_list, A, B, D, E, r))
        Fr.append(x.get())

    pool.close()
    pool.join()

    P = []
    for i in range(p):
        P.append(Fr[i])
    P = np.array(P)
    if (m * n) % p == 0:
        P = P.reshape(m, n)
    else:
        P = P[0 : m * n].reshape(m, n)

    return P
