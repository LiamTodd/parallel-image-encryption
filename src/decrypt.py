from src.TIDBD_diffusion import decrypt_diffusion


def decrypt(diffused_img_data, m, n):
    img_data = decrypt_diffusion(diffused_img_data, m, n)
