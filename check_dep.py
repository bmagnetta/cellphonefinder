import sys

def check():
    if sys.version_info<(3,5,0):
        sys.exit("ImportError => You need python 3.5 or later to run this script\n")
    try:
        import tensorflow
    except ImportError:
        sys.exit("ImportError => You need tensorflow: pip3 install --upgrade tensorflow")
    try:
        import numpy
    except ImportError:
        sys.exit("ImportError => You need numpy: pip3 install numpy")
    try:
        import scipy
    except ImportError:
        sys.exit("ImportError => You need scipy: pip3 install scipy")
    try:
        import cv2
    except ImportError:
        sys.exit("ImportError => You need opencv-python: pip3 install opencv-python ")
    try:
        import PIL
    except ImportError:
        sys.exit("ImportError => You need pillow: pip3 install pillow ")
    try:
        import matplotlib
    except ImportError:
        sys.exit("ImportError => You need matplotlib: pip3 install matplotlib ")
    try:
        import h5py
    except ImportError:
        sys.exit("ImportError => You need h5py: pip3 install h5py ")
    try:
        import keras
    except ImportError:
        sys.exit("ImportError => You need keras: pip3 install keras ")
    try:
        import imageai.Detection
    except ImportError:
        sys.exit("ImportError => You need imagai: pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl")


if __name__ == "__main__":
    check()

