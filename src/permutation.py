import numpy as np


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
