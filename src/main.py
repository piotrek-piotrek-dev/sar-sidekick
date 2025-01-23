import sys


if __name__ == '__main__':
    print(r"Starting main script")
    from utils import setup
    if not setup.install_dependencies():
        sys.exit(1)