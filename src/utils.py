import os
import random
import sys

from PIL import Image

from src.constants import DEFAULT_IMG


def parse_args():
    show_images = False
    file_path = DEFAULT_IMG
    if len(sys.argv) > 1:
        show_images = sys.argv[1] == "1"
        if len(sys.argv) > 2:
            file_path = f"./img/{sys.argv[2]}"
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return 1
    return show_images, file_path


def logistic_map(x, mu):
    return mu * x * (1 - x)


def generate_secret_key(length):
    key = ""
    for _ in range(length):
        key += str(random.randint(0, 1))
    return key


def handle_output(
    image, permutated_img_data, diffused_img_data, decrypted_img_data, show_images=False
):
    # Convert images to "RGB" mode
    permuted_image = Image.fromarray(permutated_img_data).convert("RGB")
    diffused_image = Image.fromarray(diffused_img_data).convert("RGB")
    decrypted_image = Image.fromarray(decrypted_img_data).convert("RGB")

    # Define file paths for saving
    permuted_file_path = "img/permuted_image.png"
    diffused_file_path = "img/diffused_image.png"
    decrypted_file_path = "img/decrypted_image.png"

    # Save the images
    permuted_image.save(permuted_file_path)
    diffused_image.save(diffused_file_path)
    decrypted_image.save(decrypted_file_path)

    if show_images:
        image.show()
        permuted_image.show()
        diffused_image.show()
        decrypted_image.show()
