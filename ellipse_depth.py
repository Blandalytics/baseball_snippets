"""Overlap depth of per-group 1-SD location ellipses.

The ellipses are the ones ``confidence_ellipse`` draws: the covariance ellipse
of (x, y), scaled by ``n_std``.  Depth at a point is how many of them cover it.

    from ellipse_depth import ellipse_depth
    import statfast

    df = statfast.pitcher_season("Paul Skenes", 2026)
    r = ellipse_depth(df)
    print(r.report())
    r.per_group.mean_depth      # per pitch type
    r.weighted_mean_depth       # usage-weighted across types
    r.union_mean_depth          # over the union of every ellipse

Nothing here is specific to seven pitch types or to pitch data: ``group``,
``x`` and ``y`` name the columns, and any number of groups works.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["EllipseDepth", "ellipse_depth", "group_ellipse"]


@dataclass(frozen=True)
class Ellipse2D:
    """The covariance ellipse of a group, in the drawn parameterisation.

    Points on it are ``mean + scale * R45 @ (rx cos t, ry sin t)``, matching
    ``Affine2D().rotate_deg(45).scale(*scale).translate(*mean)`` applied to an
    ellipse of radii ``(rx, ry) = (sqrt(1 + r), sqrt(1 - r))``.
    """

    mean: tuple[float, float]
    scale: tuple[float, float]
    radii: tuple[float, float]
    pearson: float

    @property
    def area(self) -> float:
        """Exact area: pi * n_std^2 * sqrt(det(cov))."""
        return np.pi * self.scale[0] * self.scale[1] * self.radii[0] * self.radii[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax).

        The 45-degree rotation makes the half-extents exactly ``scale``:
        sqrt((rx cos45)^2 + (ry sin45)^2) = sqrt((1 + r + 1 - r) / 2) = 1.
        """
        (mx, my), (sx, sy) = self.mean, self.scale
        return mx - sx, mx + sx, my - sy, my + sy

    def contains(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Boolean array of which (gx, gy) fall inside, by inverting the draw
        transform: undo the translate, the scale, then the 45-degree rotation."""
        (mx, my), (sx, sy), (rx, ry) = self.mean, self.scale, self.radii
        u, v = (gx - mx) / sx, (gy - my) / sy
        a = -np.pi / 4
        p = u * np.cos(a) - v * np.sin(a)
        q = u * np.sin(a) + v * np.cos(a)
        return (p / rx) ** 2 + (q / ry) ** 2 <= 1.0


def group_ellipse(x, y, n_std: float = 1.0) -> Ellipse2D | None:
    """The 1-SD ellipse of x and y, or None if it would be degenerate.

    Degenerate means fewer than three points, a zero-variance axis, or perfectly
    collinear points - each gives an ellipse of zero area, which no amount of
    grid resolution can resolve.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return None
    r = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    if not np.isfinite(r) or abs(r) >= 1.0 - 1e-12:
        return None
    return Ellipse2D(
        mean=(float(x.mean()), float(y.mean())),
        scale=(float(np.sqrt(cov[0, 0]) * n_std), float(np.sqrt(cov[1, 1]) * n_std)),
        radii=(float(np.sqrt(1 + r)), float(np.sqrt(1 - r))),
        pearson=r,
    )


@dataclass
class EllipseDepth:
    """Depth results.  ``per_group`` is one row per group, ordered by usage.

    Columns: ``n`` (rows), ``usage`` (share of all rows, including any skipped
    groups), ``weight`` (usage renormalised over the kept groups, sums to 1),
    ``area`` (exact, sq units), ``mean_depth`` (mean ellipses covering a point
    inside this group's own ellipse, itself included), ``weighted_depth``
    (``weight * mean_depth``, so the column sums to ``weighted_mean_depth``),
    and ``grid_area`` / ``area_error`` as a resolution check.
    """

    per_group: pd.DataFrame
    weighted_mean_depth: float
    union_mean_depth: float
    union_area: float
    max_depth: int
    depth_area: pd.Series
    skipped: dict[object, str] = field(default_factory=dict)
    grid: tuple[int, int] = (0, 0)

    def report(self) -> str:
        cols = ["n", "usage", "weight", "area", "mean_depth", "weighted_depth"]
        out = [self.per_group[cols].round(4).to_string(),
               "",
               f"usage-weighted mean_depth : {self.weighted_mean_depth:.3f}",
               f"union mean depth          : {self.union_mean_depth:.3f}",
               f"union area                : {self.union_area:.3f}",
               f"max depth                 : {self.max_depth}",
               f"grid                      : {self.grid[0]} x {self.grid[1]}",
               f"worst area error          : "
               f"{self.per_group.area_error.abs().max() * 100:.2f}%"]
        if self.skipped:
            out.append("skipped: " + ", ".join(f"{k} ({v})"
                                               for k, v in self.skipped.items()))
        return "\n".join(out)


def ellipse_depth(df: pd.DataFrame, *, group: str = "pitch_type",
                  x: str = "plate_x", y: str = "plate_z", n_std: float = 1.0,
                  grid: int = 1400, pad: float = 0.05,
                  min_rows: int = 3) -> EllipseDepth:
    """Overlap depth of every group's ``n_std`` ellipse.

    Parameters
    ----------
    df : rows to group; needs the ``group``, ``x`` and ``y`` columns.
    group, x, y : column names.
    n_std : ellipse size, in standard deviations.
    grid : samples along the wider axis; cells are kept square.  Depth areas
        are sampled, so raise this if ``area_error`` is too large for you.
    pad : fraction of the bounding box to pad the grid by, so no ellipse is
        clipped by the sampling window.
    min_rows : groups with fewer rows than this are skipped as degenerate.

    Returns
    -------
    EllipseDepth
    """
    for col in (group, x, y):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in the frame")

    data = df[[group, x, y]].dropna(subset=[x, y])
    if data.empty:
        raise ValueError("no rows with finite coordinates")
    total = len(data)

    ellipses: dict[object, Ellipse2D] = {}
    counts: dict[object, int] = {}
    skipped: dict[object, str] = {}
    for key, sub in data.groupby(group, observed=True, sort=False):
        counts[key] = len(sub)
        if len(sub) < min_rows:
            skipped[key] = f"{len(sub)} rows < min_rows={min_rows}"
            continue
        e = group_ellipse(sub[x], sub[y], n_std)
        if e is None:
            skipped[key] = "degenerate covariance"
        else:
            ellipses[key] = e
    if not ellipses:
        raise ValueError("no group produced a non-degenerate ellipse")

    # a grid over every ellipse's bounding box, with square cells
    bounds = np.array([e.bounds for e in ellipses.values()])
    x0, x1 = bounds[:, 0].min(), bounds[:, 1].max()
    y0, y1 = bounds[:, 2].min(), bounds[:, 3].max()
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    w, h = x1 - x0, y1 - y0
    nx = int(grid) if w >= h else max(2, round(grid * w / h))
    ny = max(2, round(nx * h / w))
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    cell = (w / nx) * (h / ny)

    # pass one: total depth.  Only one mask is held at a time, so the cost does
    # not grow with the number of groups.
    depth = np.zeros(gx.shape, dtype=np.int16)
    for e in ellipses.values():
        depth += e.contains(gx, gy)

    keys = sorted(ellipses, key=lambda k: -counts[k])
    kept = sum(counts[k] for k in keys)
    rows = []
    for key in keys:
        e = ellipses[key]
        inside = e.contains(gx, gy)
        rows.append({
            group: key,
            "n": counts[key],
            "usage": counts[key] / total,
            "weight": counts[key] / kept,
            "area": e.area,
            "grid_area": inside.sum() * cell,
            "mean_depth": float(depth[inside].mean()) if inside.any() else np.nan,
            "pearson": e.pearson,
        })

    per = pd.DataFrame(rows).set_index(group)
    per["area_error"] = per.grid_area / per.area - 1
    per["weighted_depth"] = per.weight * per.mean_depth
    per = per[["n", "usage", "weight", "area", "grid_area", "area_error",
               "mean_depth", "weighted_depth", "pearson"]]

    covered = depth >= 1
    counts_by_depth = np.bincount(depth.ravel(), minlength=int(depth.max()) + 1)
    return EllipseDepth(
        per_group=per,
        weighted_mean_depth=float(per.weighted_depth.sum()),
        union_mean_depth=float(depth[covered].mean()),
        union_area=float(covered.sum() * cell),
        max_depth=int(depth.max()),
        depth_area=pd.Series(counts_by_depth[1:] * cell,
                             index=pd.RangeIndex(1, len(counts_by_depth),
                                                 name="depth"),
                             name="area"),
        skipped=skipped,
        grid=(nx, ny),
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    df = pd.read_parquet("skenes_2026.parquet")
    r = ellipse_depth(df)
    print(r.report())
    print("\narea by depth:")
    print(r.depth_area.round(3).to_string())
