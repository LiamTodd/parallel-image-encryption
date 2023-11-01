from src.permutation import decrypt_permutation
from src.TIDBD_diffusion import decrypt_diffusion


def decrypt(cipher, M, N, H, S, m, n, p, A, B, D, E, GD1, GD2):
    """
    This function decrypts the cipher of an image which has been encrypted using TIDBD.
    It requires the lattices from the CML used to encrypt the image, as well as the encrypted image (cipher).
    """
    img_data = decrypt_diffusion(cipher, m, n, p, A, B, D, E, GD1, GD2)
    img_data = decrypt_permutation(img_data, M, N, H, S)
    return img_data
