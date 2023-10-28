import numpy as np
from PIL import Image

from src.constants import SECRET_KEY_BITS, THREADS
from src.coupled_map_logistic_lattices import (
    generate_coupled_map_logistic_lattices,
    process_cmls,
)
from src.decrypt import decrypt
from src.permutation import permute_data
from src.TIDBD_diffusion import TIDBD_diffuse
from src.utils import generate_secret_key, handle_output, parse_args


def main():
    show_images, file_path = parse_args()

    # convert image to greyscale
    image = Image.open(file_path).convert("L")
    img_data = np.asarray(image)
    par = 2 * THREADS + 4
    K = generate_secret_key(SECRET_KEY_BITS)
    m = img_data.shape[0]
    n = img_data.shape[1]
    # generate CMLs for permutation and diffusion
    cmls = generate_coupled_map_logistic_lattices(K, m, n, THREADS)
    M, N, H, S, A, B, D, E = process_cmls(cmls, m, n, par)
    # permute image
    permutated_img_data = permute_data(img_data, M, N, S, H)
    # diffuse image
    diffused_img_data, GD1, GD2 = TIDBD_diffuse(
        permutated_img_data, THREADS, A, B, D, E, m, n
    )
    # decrypy diffused image
    decrypted_img_data = decrypt(
        diffused_img_data, M, N, H, S, m, n, THREADS, A, B, D, E, GD1, GD2
    )

    # save and show image stages
    handle_output(
        image, permutated_img_data, diffused_img_data, decrypted_img_data, show_images
    )

    return 0


if __name__ == "__main__":
    main()
