import numpy as np


def permute_data(img_data, M, N, S, H):
    sort_column = img_data[:, np.argsort(N)]
    cirshift_d = sort_column
    for i in range(len(N)):
        cirshift_d[:, i] = np.roll(cirshift_d[:, i], S[i])
    sort_rows = cirshift_d[np.argsort(M), :]
    cirshift_r = sort_rows
    for i in range(len(M)):
        cirshift_r[i, :] = np.roll(cirshift_r[i, :], H[i])
    return cirshift_r


def decrypt_permutation(circshift_l, M, N, H, S):
    for i in range(len(M)):
        circshift_l[i, :] = np.roll(circshift_l[i, :], -H[i])
    sort_rows = circshift_l
    M_rev = np.zeros_like(M)
    M_rev[np.argsort(M)] = np.arange(len(M))
    circshift_u = sort_rows[M_rev, :]
    for i in range(len(N)):
        circshift_u[:, i] = np.roll(circshift_u[:, i], -S[i])
    sort_column = circshift_u
    N_rev = np.zeros_like(N)
    N_rev[np.argsort(N)] = np.arange(len(N))
    return sort_column[:, N_rev]
