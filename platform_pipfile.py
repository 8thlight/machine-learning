"""Prints the platform name for the current architecture
as needed by the Pipfile files under /platforms/ directory

If a new architecture is added, this script should be updated.

If a new architecture is compatible with an existing platform,
make sure it is mapped to the correct platform name.

If a new architecture is incompatible with an existing platform,
create a new platform directory, add a new Pipfile file, and
add a new entry to the if-else statement below.

Usage:
  python platform_pipfile.py

Example:
  $ python platform_pipfile.py
  x86_64

  $ python platform_pipfile.py
  arm64
"""
import platform

def main():
  machine = platform.machine()

  if machine == "arm64" or machine == "aarch64":
    print("arm64")
  elif machine == "x86_64":
    print("x86_64")
  else:
    raise Exception("Unsupported architecture: " + machine)

if __name__ == "__main__":
  main()
