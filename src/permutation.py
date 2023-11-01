import numpy as np


def permute_data(img_data, M, N, S, H):
    """
    This function permutes the image by applying a row and column based sort, as well as a vertical and horizontal cyclic-shift
    """
    # apply a sort to columns, based on lattice N
    column_sorted = img_data[:, np.argsort(N)]

    # cyclical down shift each pixel by N%height places (this is contained in S)
    down_cycled = column_sorted
    for i in range(len(N)):
        down_cycled[:, i] = np.roll(down_cycled[:, i], S[i])

    # apply a sort to rows, based lattice M
    row_sorted = down_cycled[np.argsort(M), :]

    # cyclical shift to the right by M%width (this is contained in H)
    right_cycled = row_sorted
    for i in range(len(M)):
        right_cycled[i, :] = np.roll(right_cycled[i, :], H[i])

    return right_cycled


def decrypt_permutation(permuted_img_data, M, N, H, S):
    """
    This function undoes the permutation operation defined in pemute_data()
    """
    # undo right-cycle
    left_cycled = permuted_img_data
    for i in range(len(M)):
        left_cycled[i, :] = np.roll(left_cycled[i, :], -H[i])

    # undo row sorting
    row_unsorted = left_cycled
    M_rev = np.zeros_like(M)
    M_rev[np.argsort(M)] = np.arange(len(M))

    # undo down-cycle
    up_cycled = row_unsorted[M_rev, :]
    for i in range(len(N)):
        up_cycled[:, i] = np.roll(up_cycled[:, i], -S[i])

    # undo column sorting
    column_unsorted = up_cycled
    N_rev = np.zeros_like(N)
    N_rev[np.argsort(N)] = np.arange(len(N))

    return column_unsorted[:, N_rev]
