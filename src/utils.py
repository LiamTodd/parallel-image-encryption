import os
import random
import sys

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
