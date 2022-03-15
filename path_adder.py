"""Adds this repository's modules to the Python's PATH"""
import os
import sys

import pathlib


def main():
    """
    Creates a `.pth` file under the corresponding `site-packages` dir.

    This function works in both a virtual and a native environment.
    """
    this_file_path = pathlib.Path(__file__).parent.resolve()
    src_path = os.path.join(this_file_path, "src")

    for elem in sys.path:
        if elem.endswith("site-packages"):
            site_packages_dir = elem
        elif elem == src_path:
            print(src_path + " path is already added. Aborting...")
            return

    file_path = os.path.join(site_packages_dir,
                             'extra_python_folders.pth')

    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(src_path)
        print("Added to path: " + src_path)


if __name__ == "__main__":
    main()
