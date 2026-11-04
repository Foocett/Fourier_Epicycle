import numpy as np
from scipy.fft import fft
import tkinter as tk
import matplotlib as mpl
import matplotlib.pyplot as plt
from drawFigure import draw_shape


def generate_epicycles(shape: np.ndarray) -> np.ndarray:
    pass


def main():
    print("Hello from nummethodsfinal!")
    samples = draw_shape()
    if samples is not None:
        plt.plot(samples.real, samples.imag, marker='o')

        plt.show()


if __name__ == "__main__":
    main()
