import numpy as np
from PIL import Image

from src.constants import SECRET_KEY_BITS, THREADS
from src.coupled_map_logistic_lattices import (
    generate_coupled_map_logistic_lattices,
    process_cmls,
)
from src.permutation import permutate_data
from src.TIDBD_diffusion import TIDBD_diffuse
from src.utils import generate_secret_key, parse_args


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
    diffused_img_data = TIDBD_diffuse(permutated_img_data, THREADS, A, B, D, E)

    if show_images:
        image.show()
        permuted_image = Image.fromarray(permutated_img_data)
        permuted_image.show()
        diffused_img_data.show()

    return 0


if __name__ == "__main__":
    main()
