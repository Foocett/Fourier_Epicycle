"""Draw a freehand shape and convert it to evenly spaced complex samples."""

from __future__ import annotations

from collections.abc import Sequence
import tkinter as tk
from tkinter import messagebox

import numpy as np


Point = tuple[float, float]


def resample_by_arclength(
    points: Sequence[Point],
    sample_count: int,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Return ``sample_count`` points evenly spaced along a drawn path.

    The returned array contains complex values whose real and imaginary parts
    are the x and y coordinates. Consecutive duplicate points are ignored.
    When ``closed`` is true, the last point is connected back to the first.
    """
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must be a sequence of (x, y) pairs")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must contain only finite coordinates")

    if len(coordinates) > 1:
        changes = np.any(np.diff(coordinates, axis=0) != 0, axis=1)
        coordinates = coordinates[np.r_[True, changes]]
    if closed and len(coordinates) > 1 and np.array_equal(
        coordinates[0], coordinates[-1]
    ):
        coordinates = coordinates[:-1]

    if len(coordinates) < 2:
        raise ValueError("at least two distinct points are required")

    path = (
        np.vstack((coordinates, coordinates[0]))
        if closed
        else coordinates
    )
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length == 0:
        raise ValueError("the path must have non-zero length")

    cumulative_lengths = np.r_[0.0, np.cumsum(segment_lengths)]
    target_lengths = np.linspace(
        0.0,
        total_length,
        sample_count,
        endpoint=not closed,
    )
    x = np.interp(target_lengths, cumulative_lengths, path[:, 0])
    y = np.interp(target_lengths, cumulative_lengths, path[:, 1])
    return x + 1j * y


class ShapeCanvas:
    """Small Tk canvas that records one freehand stroke."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("Draw a shape")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            background="white",
            cursor="crosshair",
        )
        self.canvas.pack(padx=8, pady=(8, 4))

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(controls, text="Done", command=self.finish).pack(
            side=tk.RIGHT
        )
        self.closed = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls,
            text="Close shape",
            variable=self.closed,
        ).pack(side=tk.RIGHT, padx=8)

        self.points: list[Point] = []
        self.result: np.ndarray | None = None
        self.cancelled = False
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.record_point)

    def start_stroke(self, event: tk.Event[tk.Misc]) -> None:
        if self.points:
            return
        point = (float(event.x), float(self.height - event.y))
        self.points.append(point)
        self.canvas.create_oval(
            event.x - 2,
            event.y - 2,
            event.x + 2,
            event.y + 2,
            fill="#1769aa",
            outline="",
        )

    def record_point(self, event: tk.Event[tk.Misc]) -> None:
        if not self.points:
            return
        point = (float(event.x), float(self.height - event.y))
        previous = self.points[-1]
        if point == previous:
            return
        self.points.append(point)
        self.canvas.create_line(
            previous[0],
            self.height - previous[1],
            event.x,
            event.y,
            fill="#1769aa",
            width=2,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

    def clear(self) -> None:
        self.canvas.delete("all")
        self.points.clear()
        self.result = None

    def finish(self) -> None:
        if len(self.points) < 2:
            messagebox.showwarning(
                "Incomplete drawing",
                "Draw a shape with at least two points first.",
                parent=self.root,
            )
            return
        self.result = np.asarray(self.points, dtype=float)
        self.root.destroy()

    def cancel(self) -> None:
        self.cancelled = True
        self.root.destroy()

    def run(self, sample_count: int) -> np.ndarray | None:
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.mainloop()
        if self.cancelled or self.result is None:
            return None
        return resample_by_arclength(
            self.result,
            sample_count=sample_count,
            closed=self.closed.get(),
        )


def draw_shape(
    sample_count: int = 1000,
    *,
    width: int = 800,
    height: int = 600,
) -> np.ndarray | None:
    """Open the drawing window and return evenly spaced complex coordinates.

    Coordinates use the canvas center as the origin, x increases rightward,
    and y increases upward. ``None`` is returned if the window is cancelled.
    """
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    canvas = ShapeCanvas(width, height)
    samples = canvas.run(sample_count)
    if samples is None:
        return None
    return samples - (width / 2) - 1j * (height / 2)


def main() -> None:
    samples = draw_shape()
    if samples is not None:
        print(samples)


if __name__ == "__main__":
    main()
