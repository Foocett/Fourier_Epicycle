# Fourier Epicycle

An interactive Python application for visualizing the Fourier decomposition of a
user-drawn two-dimensional curve as a chain of rotating epicycles.

The application represents a curve as a complex-valued signal,
`z(t) = x(t) + iy(t)`. Each Fourier coefficient contributes one rotating
vector. Connecting those vectors head-to-tail produces an epicycle chain whose
endpoint reconstructs the original curve.

## Features

- Draw one freehand curve in a Tkinter canvas.
- Optionally close the curve by connecting its final point to its first point.
- Resample the drawing at uniform arc-length intervals.
- Compute the strongest complex Fourier components with SciPy's FFT.
- Animate the epicycle chain and its traced endpoint.
- Adjust the number of included epicycles with an `N` slider.
- Inspect reconstruction error with an RMSE-versus-`N` plot.
- Inspect the signed complex Fourier spectrum and the currently included
  frequencies.
- Pause and resume the animation, adjust playback speed, and enable looping.

## Requirements

- Python 3.14 or newer
- Tkinter
- A desktop environment capable of opening Tk and Matplotlib windows

The dependencies are declared in [`pyproject.toml`](./pyproject.toml):
NumPy, SciPy, Matplotlib, and Tk.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Alternatively, create a Python 3.14+ virtual environment and install the
project dependencies from `pyproject.toml` with your preferred package manager.
Tkinter may need to be installed separately, depending on the operating
system.

## Usage

Start the animated visualization:

```powershell
uv run python -m src.main
```

The drawing window opens first:

1. Click and drag to draw one continuous stroke.
2. Select **Close shape** if the final point should connect to the first.
3. Select **Clear** to redraw, or **Done** to analyze the curve.
4. Close the window to cancel.

The animated view provides:

- **N**: number of epicycles used for the reconstruction.
- **Speed**: playback speed multiplier.
- **Pause/Play**: stop or resume the animation.
- **Loop**: restart the animation automatically when it reaches the end.

For a non-animated reconstruction and analysis view, use:

```powershell
uv run python -m src.main --quick
```

The quick view permits `N = 0`, while the animated epicycle view requires at
least one epicycle.

## How it works

The drawing is converted into complex samples
`z_k = x_k + iy_k`. By default, [`draw_shape`](./src/drawFigure.py) captures
the stroke and resamples it to 1,000 points at approximately uniform
arc-length intervals. For a sample count `M`, the Fourier coefficients are
computed as:

```text
c_n = (1 / M) * sum(z_k * exp(-2*pi*i*n*k/M))
```

The application selects the `N` coefficients with the greatest magnitudes. At
normalized time `t`, each selected component is:

```text
c_n * exp(2*pi*i*n*t)
```

The sum of those components is the reconstructed point. The displayed RMSE is
the error between the sampled input curve and that reconstruction. Because the
Fourier representation is periodic, an open stroke is treated as wrapping from
its final sample back to its first sample.

The core numerical functions are available from
[`src/main.py`](./src/main.py):

- `generate_epicycles(shape, n)` returns the strongest components and their
  magnitudes, periods, phases, and signed frequencies.
- `recreate_shape(shape, n)` evaluates the reconstruction at the original
  sample positions.

## Project layout

```text
.
├── src/
│   ├── drawFigure.py  # Tk drawing canvas and arc-length resampling
│   └── main.py        # Fourier analysis, plots, and animation
├── Project Proposal.pdf
├── pyproject.toml
└── LICENSE.md
```

## Proposal status

The current code implements the main visualization described in the proposal,
including complex-plane reconstruction, an epicycle animation, RMSE analysis,
and a Fourier spectrum. The following proposal items are not implemented yet:

- Built-in preset curves.
- A directly implemented DFT and analytical coefficient verification.
- Timeline scrubbing and a restart control.
- Visibility controls for the epicycles and original curve.
- A configurable `N = 0` to `100` harmonic-index interface independent of the
  available sample count.

These are suitable directions for future development. In particular, the
current implementation uses SciPy's optimized `fft` rather than the proposal's
planned educational direct DFT.

## License

This project is distributed under the MIT License. See
[`LICENSE.md`](./LICENSE.md).