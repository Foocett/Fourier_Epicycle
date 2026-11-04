import argparse

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from drawFigure import draw_shape


EPICYCLE_DTYPE = np.dtype(
    [
        ("magnitude", np.float64),
        ("period", np.float64),
        ("phase", np.float64),
        ("frequency", np.float64),
    ]
)


def generate_epicycles(shape: np.ndarray, n: int) -> np.ndarray:
    """Generate the ``n`` strongest Fourier epicycles for a complex curve.

    ``shape`` must contain complex coordinates sampled uniformly over one
    traversal of the curve. The returned structured array is sorted from
    largest to smallest magnitude and contains:

    - ``magnitude``: radius of the rotating vector
    - ``period``: positive period in traversal units
    - ``phase``: initial angle in radians
    - ``frequency``: signed rotations per traversal, used for direction

    At normalized time ``t`` in ``[0, 1]``, an epicycle is
    ``magnitude * exp(1j * (2*pi*frequency*t + phase))``.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")

    samples = np.asarray(shape, dtype=np.complex128)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError(
            "shape must be a one-dimensional array with at least 2 samples")
    if not np.isfinite(samples).all():
        raise ValueError("shape must contain only finite complex coordinates")
    if n > samples.size:
        raise ValueError("n cannot exceed the number of shape samples")

    coefficients = fft(samples) / samples.size
    frequencies = fftfreq(samples.size) * samples.size
    magnitudes = np.abs(coefficients)
    strongest = np.argsort(magnitudes)[::-1][:n]

    selected_frequencies = frequencies[strongest]
    periods = np.full(n, np.inf, dtype=float)
    nonzero = selected_frequencies != 0
    periods[nonzero] = 1.0 / np.abs(selected_frequencies[nonzero])

    epicycles = np.empty(n, dtype=EPICYCLE_DTYPE)
    epicycles["magnitude"] = magnitudes[strongest]
    epicycles["period"] = periods
    epicycles["phase"] = np.angle(coefficients[strongest])
    epicycles["frequency"] = selected_frequencies
    return epicycles


def recreate_shape(shape: np.ndarray, n: int) -> np.ndarray:
    """Recreate a shape from its ``n`` strongest epicycles."""
    epicycles = generate_epicycles(shape, n)
    sample_times = np.arange(shape.size, dtype=float) / shape.size
    return np.sum(
        epicycles["magnitude"][:, np.newaxis]
        * np.exp(
            1j
            * (
                2 * np.pi * epicycles["frequency"][:, np.newaxis]
                * sample_times
                + epicycles["phase"][:, np.newaxis]
            )
        ),
        axis=0,
    )


def quick_render() -> None:
    samples = draw_shape()
    if samples is None:
        return

    maximum_n = samples.size
    initial_n = min(20, maximum_n)
    reconstructed = recreate_shape(samples, initial_n)

    figure, axes = plt.subplots()
    figure.subplots_adjust(bottom=0.2)
    axes.plot(
        samples.real,
        samples.imag,
        color="0.7",
        label="Drawn shape",
    )
    reconstructed_line, = axes.plot(
        reconstructed.real,
        reconstructed.imag,
        color="tab:blue",
        label=f"Recreated ({initial_n} epicycles)",
    )
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_title("Fourier epicycle reconstruction")
    axes.legend()

    slider_axes = figure.add_axes((0.15, 0.06, 0.7, 0.04))
    n_slider = Slider(
        slider_axes,
        "N",
        valmin=1,
        valmax=maximum_n,
        valinit=initial_n,
        valstep=1,
    )

    def update(value: float) -> None:
        n = int(value)
        recreated = recreate_shape(samples, n)
        reconstructed_line.set_data(recreated.real, recreated.imag)
        reconstructed_line.set_label(f"Recreated ({n} epicycles)")
        axes.legend()
        figure.canvas.draw_idle()

    n_slider.on_changed(update)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="render the non-animated reconstruction test",
    )
    arguments = parser.parse_args()

    if arguments.quick:
        quick_render()
        return

    # Animation will be implemented here.


if __name__ == "__main__":
    main()
