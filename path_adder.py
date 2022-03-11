import os
import sys

import pathlib

def main():
    for elem in sys.path:
        this_file_path = pathlib.Path(__file__).parent.resolve()
        src_path = os.path.join(this_file_path, "src")

        if elem.endswith("site-packages"):
            site_packages_dir = elem
        elif elem == src_path:
            print(src_path + " path is already added. Aborting...")
            return
    
    file_name = 'extra_python_folders.pth'
    with open(os.path.join(site_packages_dir, file_name), 'w') as f:
        f.write(src_path)
        print("Added to conda path: " + src_path)

if __name__ == "__main__":
    main()