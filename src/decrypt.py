from src.permutation import decrypt_permutation
from src.TIDBD_diffusion import decrypt_diffusion


def decrypt(diffused_img_data, M, N, H, S, m, n, p, A, B, D, E, GD1, GD2):
    img_data = decrypt_diffusion(diffused_img_data, m, n, p, A, B, D, E, GD1, GD2)
    img_data = decrypt_permutation(img_data, M, N, H, S)
    return img_data
