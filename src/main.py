import argparse

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, CheckButtons, Slider
from src.drawFigure import draw_shape


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
    if n == 0:
        return np.zeros_like(shape, dtype=np.complex128)
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

    maximum_n = min(100, samples.size)
    initial_n = min(20, maximum_n)
    reconstructed = recreate_shape(samples, initial_n)

    coefficients = fft(samples) / samples.size
    frequencies = (fftfreq(samples.size) * samples.size).astype(int)
    analysis_n = np.arange(maximum_n + 1)
    rmse = np.array(
        [
            np.sqrt(np.mean(np.abs(samples - recreate_shape(samples, n)) ** 2))
            for n in analysis_n
        ]
    )
    plotted_rmse = np.maximum(rmse, np.finfo(float).tiny)

    figure = plt.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=(2, 1))
    axes = figure.add_subplot(grid[0, :])
    error_axes = figure.add_subplot(grid[1, 0])
    spectrum_axes = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(bottom=0.2, hspace=0.35)
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
        label=f"Recreated ({initial_n})",
    )
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_title(f"Fourier epicycle reconstruction ({initial_n})")
    axes.legend()

    error_axes.plot(analysis_n, plotted_rmse, color="tab:red")
    error_marker, = error_axes.plot(
        initial_n,
        plotted_rmse[initial_n],
        "o",
        color="black",
        label=f"N = {initial_n}",
    )
    error_axes.set_title("RMSE by number of epicycles")
    error_axes.set_xlabel("N")
    error_axes.set_ylabel("RMSE")
    error_axes.set_yscale("log")
    error_axes.legend()

    order = np.argsort(frequencies)
    spectrum_stems = spectrum_axes.stem(
        frequencies[order],
        np.maximum(np.abs(coefficients[order]), np.finfo(float).tiny),
        linefmt="-",
        markerfmt=".",
        basefmt=" ",
        label="All coefficients",
    )
    spectrum_stems.stemlines.set_color("0.7")
    included = generate_epicycles(samples, initial_n)
    included_frequencies = included["frequency"].astype(int)
    included_mask = np.isin(frequencies, included_frequencies)
    included_scatter = spectrum_axes.scatter(
        frequencies[included_mask],
        np.maximum(
            np.abs(coefficients[included_mask]),
            np.finfo(float).tiny,
        ),
        color="tab:orange",
        zorder=3,
        label="Included",
    )
    spectrum_axes.set_title("Complex Fourier spectrum")
    spectrum_axes.set_xlabel("Signed frequency")
    spectrum_axes.set_ylabel(r"$|c_n|$")
    spectrum_axes.set_yscale("log")
    spectrum_axes.legend()

    slider_axes = figure.add_axes((0.15, 0.06, 0.7, 0.04))
    n_slider = Slider(
        slider_axes,
        "N",
        valmin=0,
        valmax=maximum_n,
        valinit=initial_n,
        valstep=1,
    )

    def update(value: float) -> None:
        n = int(value)
        recreated = recreate_shape(samples, n)
        reconstructed_line.set_data(recreated.real, recreated.imag)
        reconstructed_line.set_label(f"Recreated ({n})")
        axes.set_title(f"Fourier epicycle reconstruction ({n})")
        axes.legend()
        error_marker.set_data([n], [plotted_rmse[n]])
        error_marker.set_label(f"N = {n}")
        error_axes.legend()
        selected = generate_epicycles(samples, n) if n else np.empty(
            0, dtype=EPICYCLE_DTYPE
        )
        selected_frequencies = selected["frequency"].astype(int)
        selected_mask = np.isin(frequencies, selected_frequencies)
        included_scatter.set_offsets(
            np.column_stack(
                (
                    frequencies[selected_mask],
                    np.maximum(
                        np.abs(coefficients[selected_mask]),
                        np.finfo(float).tiny,
                    ),
                )
            )
        )
        figure.canvas.draw_idle()

    n_slider.on_changed(update)
    plt.show()


def animate_shape() -> None:
    samples = draw_shape()
    if samples is None:
        return

    sample_count = samples.size
    maximum_n = min(100, sample_count)
    initial_n = min(20, maximum_n)
    epicycles = generate_epicycles(samples, initial_n)
    frame_position = 0.0
    playing = True
    looping = False
    trace: list[complex] = []

    coefficients = fft(samples) / samples.size
    frequencies = (fftfreq(sample_count) * sample_count).astype(int)
    analysis_n = np.arange(maximum_n + 1)
    rmse = np.array(
        [
            np.sqrt(
                np.mean(np.abs(samples - recreate_shape(samples, n)) ** 2)
            )
            for n in analysis_n
        ]
    )
    plotted_rmse = np.maximum(rmse, np.finfo(float).tiny)

    figure = plt.figure(figsize=(12, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(2, 1))
    axes = figure.add_subplot(grid[0, :])
    error_axes = figure.add_subplot(grid[1, 0])
    spectrum_axes = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(bottom=0.27, hspace=0.4)
    axes.plot(samples.real, samples.imag, color="0.85", label="Drawn shape")
    trace_line, = axes.plot([], [], color="tab:blue", label="Trace")
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_title("Fourier epicycles")
    axes.legend()

    extent = max(
        float(np.max(np.abs(samples.real))),
        float(np.max(np.abs(samples.imag))),
        float(np.sum(epicycles["magnitude"])),
        1.0,
    )
    padding = extent * 0.1
    axes.set_xlim(-extent - padding, extent + padding)
    axes.set_ylim(-extent - padding, extent + padding)

    error_axes.plot(analysis_n, plotted_rmse, color="tab:red")
    error_marker, = error_axes.plot(
        initial_n,
        plotted_rmse[initial_n],
        "o",
        color="black",
        label=f"N = {initial_n}",
    )
    error_axes.set_title("RMSE by number of epicycles")
    error_axes.set_xlabel("N")
    error_axes.set_ylabel("RMSE")
    error_axes.set_yscale("log")
    error_axes.legend()

    order = np.argsort(frequencies)
    spectrum_stems = spectrum_axes.stem(
        frequencies[order],
        np.maximum(np.abs(coefficients[order]), np.finfo(float).tiny),
        linefmt="-",
        markerfmt=".",
        basefmt=" ",
        label="All coefficients",
    )
    spectrum_stems.stemlines.set_color("0.7")
    included = generate_epicycles(samples, initial_n)
    included_frequencies = included["frequency"].astype(int)
    included_mask = np.isin(frequencies, included_frequencies)
    included_scatter = spectrum_axes.scatter(
        frequencies[included_mask],
        np.maximum(
            np.abs(coefficients[included_mask]),
            np.finfo(float).tiny,
        ),
        color="tab:orange",
        zorder=3,
        label="Included",
    )
    spectrum_axes.set_title("Complex Fourier spectrum")
    spectrum_axes.set_xlabel("Signed frequency")
    spectrum_axes.set_ylabel(r"$|c_n|$")
    spectrum_axes.set_yscale("log")
    spectrum_axes.legend()

    circle_artists: list[Circle] = []
    vector_artists = []

    def configure_epicycles(new_n: int) -> None:
        nonlocal epicycles
        for artist in circle_artists + vector_artists:
            artist.remove()
        circle_artists.clear()
        vector_artists.clear()
        epicycles = generate_epicycles(samples, new_n)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, new_n))
        for epicycle, color in zip(epicycles, colors):
            circle = Circle((0, 0), epicycle["magnitude"], color=color,
                            fill=False, alpha=0.45)
            vector, = axes.plot([], [], color=color, linewidth=1)
            axes.add_patch(circle)
            circle_artists.append(circle)
            vector_artists.append(vector)
        axes.set_title(f"Fourier epicycles ({new_n})")

    def render_frame(frame: int) -> None:
        time = frame / sample_count
        endpoint = 0j
        for epicycle, circle, vector in zip(
            epicycles, circle_artists, vector_artists
        ):
            start = endpoint
            endpoint += epicycle["magnitude"] * np.exp(
                1j * (
                    2 * np.pi * epicycle["frequency"] * time
                    + epicycle["phase"]
                )
            )
            circle.center = (start.real, start.imag)
            vector.set_data(
                [start.real, endpoint.real],
                [start.imag, endpoint.imag],
            )
        trace.append(endpoint)
        trace_array = np.asarray(trace, dtype=np.complex128)
        trace_line.set_data(trace_array.real, trace_array.imag)
        figure.canvas.draw_idle()

    configure_epicycles(initial_n)
    render_frame(0)

    n_slider_axes = figure.add_axes((0.15, 0.12, 0.7, 0.04))
    n_slider = Slider(
        n_slider_axes,
        "N",
        valmin=1,
        valmax=maximum_n,
        valinit=initial_n,
        valstep=1,
    )
    speed_slider_axes = figure.add_axes((0.15, 0.06, 0.7, 0.04))
    speed_slider = Slider(
        speed_slider_axes,
        "Speed",
        valmin=0.1,
        valmax=5.0,
        valinit=1.0,
    )
    play_button_axes = figure.add_axes((0.02, 0.06, 0.1, 0.06))
    play_button = Button(play_button_axes, "Pause")
    loop_button_axes = figure.add_axes((0.86, 0.06, 0.12, 0.06))
    loop_button = CheckButtons(loop_button_axes, ["Loop"], [False])
    timer = figure.canvas.new_timer(interval=30)

    def advance() -> None:
        nonlocal frame_position, playing
        frame_position += speed_slider.val
        if frame_position >= sample_count:
            if looping:
                frame_position %= sample_count
                trace.clear()
            else:
                frame_position = sample_count - 1
                playing = False
                timer.stop()
                play_button.label.set_text("Play")
        render_frame(int(frame_position))

    def toggle_play(_event) -> None:
        nonlocal playing
        del _event
        playing = not playing
        if playing:
            timer.start()
            play_button.label.set_text("Pause")
        else:
            timer.stop()
            play_button.label.set_text("Play")
        figure.canvas.draw_idle()

    def toggle_loop(_label) -> None:
        nonlocal looping
        del _label
        looping = not looping

    def update_n(value: float) -> None:
        nonlocal frame_position
        n = int(value)
        was_playing = playing
        timer.stop()
        frame_position = 0.0
        trace.clear()
        configure_epicycles(n)
        render_frame(0)
        axes.set_title(f"Fourier epicycles ({n})")
        error_marker.set_data([n], [plotted_rmse[n]])
        error_marker.set_label(f"N = {n}")
        error_axes.legend()
        selected = generate_epicycles(samples, n)
        selected_frequencies = selected["frequency"].astype(int)
        selected_mask = np.isin(frequencies, selected_frequencies)
        included_scatter.set_offsets(
            np.column_stack(
                (
                    frequencies[selected_mask],
                    np.maximum(
                        np.abs(coefficients[selected_mask]),
                        np.finfo(float).tiny,
                    ),
                )
            )
        )
        if was_playing:
            timer.start()

    timer.add_callback(advance)
    play_button.on_clicked(toggle_play)
    loop_button.on_clicked(toggle_loop)
    n_slider.on_changed(update_n)
    timer.start()
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

    animate_shape()


if __name__ == "__main__":
    main()
