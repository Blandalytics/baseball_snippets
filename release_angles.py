"""Release-angle ellipses for any pitcher: three encodings plus the loop.

Generalises skenes_2026_angles.py.  Layout is derived from the data - the
window, the frame and every pitch name's spot are computed from the ellipse
cluster - so a different arsenal lays itself out without hand-tuning.

    python release_angles.py "Roki Sasaki" 2026

Writes <slug>_<year>_angles_{type,count,share}.png and <slug>_<year>_angles.gif,
and prints the overlap-depth table.

Data: statfast.py (github.com/Blandalytics/statcast_scraper) -> MLB StatsAPI.
"""
import argparse
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects, transforms
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap, Normalize)
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, ".")
import statfast
from ellipse_depth import group_ellipse

FPS, HOLD, TRANS, DPI_GIF, DPI_PNG = 20, 72, 14, 80, 200
STATES = 4                  # outline, count, share, concentration

NAMES = {"FF": "Four-Seam", "SI": "Sinker", "CH": "Changeup", "FS": "Splitter",
         "SL": "Slider", "ST": "Sweeper", "CU": "Curveball", "FC": "Cutter",
         "KC": "Knuckle Curve", "SV": "Slurve", "FO": "Forkball",
         "KN": "Knuckleball", "EP": "Eephus", "SC": "Screwball",
         "CS": "Slow Curve", "FA": "Fastball", "FT": "Two-Seam",
         "GY": "Gyroball", "PO": "Pitchout"}
# One colour per pitch code.  Codes that share a colour do so deliberately -
# splitter/forkball, the curveball family, changeup/screwball - so no clash
# resolution here: a repeat is the intended reading.
HUES = {
    'FF': '#FF6683',
    'SI': '#F2B24B',
    'FS': '#83D6FF',
    'FO': '#83D6FF',
    'FC': '#C59C9C',
    'SL': '#CE66FF',
    'ST': '#FFAAF7',
    'CU': '#339cff',
    'CS': '#2A98FF',
    'SV': '#2A98FF',
    'CH': '#6DE95D',
    'SC': '#6DE95D',
    'KN': '#999999',
    'UN': '#999999',
}

FONT_FAMILY = "DM Sans"
FONT_VAR_URL = ("https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/"
                "DMSans%5Bopsz%2Cwght%5D.ttf")
FONT_VAR_FILE = "DMSans[opsz,wght].ttf"

WATERMARK_URL = ("https://res.cloudinary.com/dduabusaf/image/upload/v1772839288/"
                 "PitcherList_Stats_watermark_with_logo_k9e3xa.webp")
WATERMARK_FILE = "pitcherlist_watermark.webp"
WATERMARK_W = 0.25          # fraction of figure width

SURFACE, TITLE, RULE = "#262940", "#72CBFD", "#4a4d63"
INK = INK2 = INK3 = "#ffffff"
# Sequential ramp for a dark surface.  Every scale is mapped the same way: the
# lowest value present is the background itself, the highest is white.
RAMP_ENDS = ["#262940", "#ffffff"]

# The axes are equal-aspect, so the data window sets the figure shape.  The
# frame is the ellipse cluster plus a margin wide enough to hold a pitch name,
# since the names sit inside it; around that go a thin pad each side, a band
# underneath for the legend, and whatever slack the aspect clamp adds.
FRAME_PAD = (0.26, 0.20)    # margin round the cluster, as a fraction of span
SIDE_PAD, TOP_PAD, BOT_PAD = 0.07, 0.04, 0.24
GRIDC = "#3a3d55"           # degree grid, one step off the ground

# Figure geometry.  The axes box is a fixed slice of the figure and the axes
# are equal-aspect, so the box's shape fixes the data window's shape too -
# there is exactly one window aspect that fills the box, and window() works to
# it rather than choosing one.  The box is square, which is what makes the two
# axis limits span the same number of degrees: the window is as tall as it is
# wide whatever the arsenal, so a degree of HRA and a degree of VRA are the
# same length and the same fraction of the chart.
FIG_W, FIG_ASPECT = 11.6, 1.0       # inches across, and width : height
L, B, W_FRAC, TOP = 0.035, 0.052, 0.93, 0.875   # chrome: rule, title, footers
RULE_Y = 0.888              # the header band is 1.0 down to here
FOOT_Y = 0.016              # baseline the footer note and word mark share
FIG_H = FIG_W / FIG_ASPECT
# Height comes from the band left between header and footer; width matches it
# in inches and the box is centred.  Only the plotting area insets - the rule
# and the footers keep the wider chrome margins.
AX_H = TOP - B
AX_W = AX_H * FIG_H / FIG_W
AX_L = (1 - AX_W) / 2
AXES_RATIO = AX_W * FIG_W / (AX_H * FIG_H)      # 1.0, by construction
# A name may not touch an ellipse, so the search runs over spots that land
# clear of every fill; among those, what one costs is an earlier name under it,
# poking out of the frame, and distance from its own ellipse.  W_INK only picks
# the least-bad spot in the fallback, for a cluster that leaves nothing clear.
W_INK, W_LAP, W_OUT, W_TET = 5.0, 8.0, 6.0, 1.3
LABEL_PT = 18               # the names' type size, in points
# Fractions of a name's own height: how far an ellipse is held off the glyphs,
# where the leader starts outside the name, the stub too short to be worth
# drawing, and the ladder of steps outward the search walks for clear surface.
LABEL_CLEAR, LEADER_GAP, LEADER_MIN = 1.13, 0.28, 0.30
# The rungs start well off the outline: among spots that are clear, the
# nearest wins, so where the ladder starts is what sets the standoff.
LADDER = (1.0, 1.9, 3.0, 4.4, 6.3, 8.8, 11.9)


def pitch_angles(dataframe):
    viz_processing_time = 0.08
    swing_time = 0.15
    total_decision_time = swing_time+viz_processing_time
    ### Physical characteristics of pitch
    ## Release Angles

    # Release Speed
    dataframe['vYs'] = -((dataframe['vy0']**2 - 2 * dataframe['ay'] * (60.5 - dataframe['release_extension'] - 50)) ** 0.5)

    # Time to plate, from start
    dataframe['pitch_time_start'] = (dataframe['vYs'] - dataframe['vy0'])/dataframe['ay']

    # Release speed, X- and Z-directions
    dataframe['vXs'] = dataframe['vx0'] - dataframe['ax'] * dataframe['pitch_time_start']
    dataframe['vZs'] = dataframe['vz0'] - dataframe['az'] * dataframe['pitch_time_start']

    # Release Angles, Horizontal, and Vertical
    dataframe['HRA'] = -1 * np.arctan(dataframe['vXs']/dataframe['vYs']) * (180/np.pi)
    dataframe['VRA'] = -1 * np.arctan(dataframe['vZs']/dataframe['vYs']) * (180/np.pi)

    # Pitch velocity (to plate) at plate
    dataframe['vYf'] = -1 * (dataframe['vy0']**2 - (2 * dataframe['ay']*(50-17/12)))**0.5
    dataframe['vYswing'] = dataframe['vYf'] - dataframe['ay'] * swing_time
    dataframe['vYdec'] = dataframe['vYf'] - dataframe['ay'] * total_decision_time

    # Pitch time in air (50ft to home plate)
    dataframe['pitch_time_50ft'] = (dataframe['vYf'] - dataframe['vy0'])/dataframe['ay']
    # Pitch velocity (vertical) at plate
    dataframe['vXf'] = dataframe['vx0'] + dataframe['ax'] * dataframe['pitch_time_50ft']
    dataframe['vXswing'] = dataframe['vXf'] - dataframe['ax'] * swing_time
    dataframe['vXdec'] = dataframe['vXf'] - dataframe['ax'] * total_decision_time

    dataframe['vZf'] = dataframe['vz0'] + dataframe['az'] * dataframe['pitch_time_50ft']
    dataframe['vZswing'] = dataframe['vZf'] - dataframe['az'] * swing_time
    dataframe['vZdec'] = dataframe['vZf'] - dataframe['az'] * total_decision_time

    # Raw horizontal angles
    dataframe['HAA'] = -1 * np.arctan(dataframe['vXf']/dataframe['vYf']) * (180/np.pi)
    dataframe['HSA'] = -1 * np.arctan(dataframe['vXswing']/dataframe['vYswing']) * (180/np.pi)
    dataframe['HDA'] = -1 * np.arctan(dataframe['vXdec']/dataframe['vYdec']) * (180/np.pi)
    # Raw vertical angles
    dataframe['VAA'] = -1 * np.arctan(dataframe['vZf']/dataframe['vYf']) * (180/np.pi)
    dataframe['VSA'] = -1 * np.arctan(dataframe['vZswing']/dataframe['vYswing']) * (180/np.pi)
    dataframe['VDA'] = -1 * np.arctan(dataframe['vZdec']/dataframe['vYdec']) * (180/np.pi)

    return dataframe[['HRA','VRA','HAA','VAA','HSA','VSA','HDA','VDA']]


def _to_lab(rgb):
    """sRGB (0-1) -> CIE L*a*b*, D65."""
    rgb = np.asarray(rgb, float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _from_lab(lab):
    """CIE L*a*b* -> sRGB (0-1), clipped."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    d = 6 / 29
    xyz = np.where(f > d, f ** 3, 3 * d ** 2 * (f - 4 / 29))
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    m = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]]))
    lin = xyz @ m.T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def lab_ramp(c0, c1, n=48):
    """Colours evenly spaced between two hexes in L*a*b*.

    Interpolating raw sRGB bunches the visible change at one end; stepping
    through L*a*b* keeps each class the same perceptual distance from the next.
    """
    ends = _to_lab([mpl.colors.to_rgb(c0), mpl.colors.to_rgb(c1)])
    t = np.linspace(0, 1, n)[:, None]
    return [tuple(c) for c in _from_lab(ends[0] + t * (ends[1] - ends[0]))]


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    """
    Draw a confidence ellipse of *x* and *y* onto *ax* based on their covariance.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        lw=2,
        **kwargs,
    )

    # Scale by the stdev of x and y, then translate to the mean.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def boundary(e, n=361):
    """The ellipse as a point set, for ringing it with candidate name spots."""
    t = np.linspace(0, 2 * np.pi, n)
    a = np.pi / 4
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = rot @ np.vstack([e.radii[0] * np.cos(t), e.radii[1] * np.sin(t)])
    return pts[0] * e.scale[0] + e.mean[0], pts[1] * e.scale[1] + e.mean[1]


def use_font():
    """Register DM Sans with matplotlib, returning the family stack to set.

    Google ships it only as a variable font, which matplotlib cannot pick
    weights out of, so cut two static instances - 400 and 700 - once and cache
    them next to the script.
    """
    from pathlib import Path
    from urllib.request import urlopen

    from matplotlib import font_manager as fm
    if any(e.name == FONT_FAMILY for e in fm.fontManager.ttflist):
        return [FONT_FAMILY, "DejaVu Sans"]
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont

        var = Path(FONT_VAR_FILE)
        if not var.exists():
            var.write_bytes(urlopen(FONT_VAR_URL, timeout=60).read())
        for wght, sub in ((400, "Regular"), (700, "Bold")):
            out = Path(f"DMSans-{sub}.ttf")
            if not out.exists():
                f = TTFont(var)
                instantiateVariableFont(f, {"wght": wght, "opsz": 14},
                                        inplace=True)
                f["OS/2"].usWeightClass = wght
                for rec in f["name"].names:
                    if rec.nameID == 1:
                        rec.string = FONT_FAMILY
                    elif rec.nameID == 2:
                        rec.string = sub
                    elif rec.nameID == 4:
                        rec.string = f"{FONT_FAMILY} {sub}"
                f.save(out)
            fm.fontManager.addfont(str(out))
        return [FONT_FAMILY, "DejaVu Sans"]
    except Exception as exc:                      # offline, or fontTools absent
        print(f"{FONT_FAMILY} unavailable ({type(exc).__name__}: {exc})")
        return ["Segoe UI", "DejaVu Sans"]


def watermark():
    """The footer mark as an RGBA array, cached locally; None if unavailable."""
    from pathlib import Path
    from urllib.request import urlopen

    from PIL import Image
    f = Path(WATERMARK_FILE)
    try:
        if not f.exists():
            f.write_bytes(urlopen(WATERMARK_URL, timeout=30).read())
        return np.asarray(Image.open(f).convert("RGBA"), float) / 255
    except Exception as exc:                      # offline, or the URL moved
        print(f"watermark skipped: {type(exc).__name__}: {exc}")
        return None


def as_date(value, season):
    """A date written any of the ways someone types one, in a given season.

    "6/17", "6-17", "06/17" and "2026-06-17" all land on the same day, so a
    segment can be given the way it is spoken about.
    """
    text = str(value).strip().replace("/", "-")
    if len(text.split("-")) == 2:
        text = f"{season}-{text}"
    return pd.Timestamp(text).normalize()


def span_text(start, end):
    """How a date segment reads, or "" for a whole season.

    Title case, because where it lands is the end of the headline's second
    line, alongside whichever overlap the chart is showing.
    """
    if start is None and end is None:
        return ""
    fmt = "%-m/%-d" if os.name != "nt" else "%#m/%#d"
    if start is None:
        return f"Through {end.strftime(fmt)}"
    if end is None:
        return f"From {start.strftime(fmt)}"
    return f"{start.strftime(fmt)} to {end.strftime(fmt)}"


def span_tag(start, end):
    """The same segment as a filename part, so the two never overwrite."""
    if start is None and end is None:
        return ""
    if start is None:
        return f"_thru{end:%m%d}"
    if end is None:
        return f"_from{start:%m%d}"
    return f"_{start:%m%d}to{end:%m%d}"


def load(pitcher, season, game_type="R", start=None, end=None):
    """The pitcher's season with HRA and VRA attached, cached to parquet.

    The cache is always the whole season, so cutting it into segments costs
    one pull however many segments are asked for.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", pitcher.lower()).strip("_")
    cache = f"{slug}_{season}.parquet"
    try:
        df = pd.read_parquet(cache)
    except (FileNotFoundError, OSError):
        df = statfast.pitcher_season(pitcher, season, game_type=game_type)
        if df.empty:
            raise SystemExit(f"no {game_type} pitches for {pitcher} in {season}")
        df.to_parquet(cache, index=False)
    if start is not None:
        df = df[df.game_date >= start]
    if end is not None:
        df = df[df.game_date <= end]
    if df.empty:
        raise SystemExit(f"no {game_type} pitches for {pitcher} in "
                         f"{season} {span_text(start, end).lower()}".rstrip())
    # the angle maths squares and square-roots velocities; float32 loses too much
    for c in ("vx0", "vy0", "vz0", "ax", "ay", "az", "release_extension"):
        df[c] = df[c].astype("float64")
    df[["HRA", "VRA"]] = pitch_angles(df)[["HRA", "VRA"]]
    return slug, df.dropna(subset=["HRA", "VRA", "pitch_type"])


def nice_ticks(lo, hi, most=6):
    """Tick values at a round step, covering (lo, hi) without crowding."""
    for step in (0.5, 1.0, 2.0, 2.5, 5.0, 10.0):
        if (hi - lo) / step <= most:
            break
    # + 0.0 folds a negative zero back to zero: ceil() hands one back
    # whenever lo sits just below the axis, and it prints as "-0"
    return np.arange(np.ceil(lo / step) * step, hi + 1e-9, step) + 0.0


def window(x0, x1, y0, y1):
    """The data window for a frame spanning (x0, x1, y0, y1).

    The axes box is square, so this is the smallest square window that still
    clears every pad - the two limits come out equal in magnitude for any
    arsenal, and the chart takes as much of the figure as that allows.
    Returns (XLIM, YLIM, dw, dh); whichever pad ends up with the slack, it
    goes below the frame, where the legend sits.
    """
    dw_min = (x1 - x0) / (1 - 2 * SIDE_PAD)
    dh_min = (y1 - y0) / (1 - TOP_PAD - BOT_PAD)
    dw = max(dw_min, AXES_RATIO * dh_min)
    dh = dw / AXES_RATIO
    xc = (x0 + x1) / 2
    XLIM = (xc - dw / 2, xc + dw / 2)
    YLIM = (y1 + TOP_PAD * dh - dh, y1 + TOP_PAD * dh)
    return XLIM, YLIM, dw, dh


def text_sizes(fig, ax, texts):
    """Each drawn label's size in data units - measured, not guessed at."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    out = {}
    for k, t in texts.items():
        bb = t.get_window_extent(rend)
        (u0, v0), (u1, v1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        out[k] = (abs(u1 - u0), abs(v1 - v0))
    return out


def label_places(ells, order, sizes, frame, depth, XLIM, YLIM, n_ang=72):
    """Where each pitch name goes, and the point its leader runs to.

    A name may not touch an ellipse - not even its own - so candidates are
    points on each outline stepped outward, along the ladder, until the name's
    box lands clear of every fill.  Clearance is tested against the depth map
    that is already built, by slicing out the cells the box covers, so it is
    exact at grid resolution rather than sampled.  Among the clear spots the
    cheapest wins: cost is how much of an earlier name it covers, how far it
    pokes out of the frame, and how far it sits from its own ellipse.  The
    busiest pitch picks first, so the names that carry the chart get the
    closest spots and the rare ones take what is left.  A cluster that leaves
    nothing clear at all falls back to the least-covered spot instead.

    Returns (spot, anchor) per pitch - the anchor being the point on its own
    outline nearest the name, which is where the leader line ends.
    """
    fx0, fx1, fy0, fy1 = frame
    ny, nx = depth.shape
    dmax = max(1, int(depth.max()))
    (gx0, gx1), (gy0, gy1) = XLIM, YLIM
    filled = depth >= 1

    def col(v):
        return int(np.clip(round((v - gx0) / (gx1 - gx0) * (nx - 1)), 0, nx - 1))

    def row(v):
        return int(np.clip(round((v - gy0) / (gy1 - gy0) * (ny - 1)), 0, ny - 1))

    def clear(cx, cy, hw, hh):
        """True when no ellipse reaches anywhere under the box."""
        return not filled[row(cy - hh):row(cy + hh) + 1,
                          col(cx - hw):col(cx + hw) + 1].any()

    def ink(cx, cy, hw, hh):
        """Mean overlap depth under the box; only the fallback needs it."""
        return depth[row(cy - hh):row(cy + hh) + 1,
                     col(cx - hw):col(cx + hw) + 1].mean() / dmax

    def outside(cx, cy, hw, hh):
        """How far the box pokes past the frame, in box widths and heights."""
        return (max(0.0, fx0 + hw - cx, cx - (fx1 - hw)) / (2 * hw)
                + max(0.0, fy0 + hh - cy, cy - (fy1 - hh)) / (2 * hh))

    def lap(cx, cy, hw, hh, placed):
        """Fraction of the box already covered by a name."""
        tot = 0.0
        for qx, qy, qhw, qhh in placed:
            w = min(cx + hw, qx + qhw) - max(cx - hw, qx - qhw)
            h = min(cy + hh, qy + qhh) - max(cy - hh, qy - qhh)
            if w > 0 and h > 0:
                tot += w * h
        return tot / (4 * hw * hh)

    placed, out = [], {}
    for p in order:
        e = ells[p]
        mx, my = e.mean
        w, h = sizes[p]
        hw, hh = w / 2, h / 2
        gap = LABEL_CLEAR * h
        bx, by = boundary(e, n_ang)
        ux, uy = bx - mx, by - my
        norm = np.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        reach = np.hypot(*e.scale)
        best, cheapest = None, np.inf
        stuck, worst = (mx, my), np.inf
        for step in LADDER:
            # the box's support function, so it clears the outline whichever
            # way the step points, plus a gap that grows with each rung
            off = gap + step * h + np.abs(ux) * hw + np.abs(uy) * hh
            for cx, cy in zip(bx + ux * off, by + uy * off):
                c = (W_LAP * lap(cx, cy, hw, hh, placed)
                     + W_OUT * outside(cx, cy, hw, hh)
                     + W_TET * float(np.hypot(cx - mx, cy - my)) / reach)
                if clear(cx, cy, hw + gap, hh + gap):
                    if c < cheapest:
                        best, cheapest = (cx, cy), c
                elif c + W_INK * ink(cx, cy, hw, hh) < worst:
                    stuck, worst = (cx, cy), c + W_INK * ink(cx, cy, hw, hh)
        cx, cy = best if best is not None else stuck
        k = int(np.hypot(bx - cx, by - cy).argmin())
        out[p] = ((cx, cy), (bx[k], by[k]))
        placed.append((cx, cy, hw, hh))
    return out


def leader(ax, spot, anchor, size, colour):
    """A hairline tying a name to its ellipse.

    It starts where the line out to the anchor leaves the name's box, so it
    never runs under the glyphs, and is dropped when the name already sits
    close enough that the line would be a stub rather than a connector.
    """
    (cx, cy), (tx, ty) = spot, anchor
    hw = size[0] / 2 + LEADER_GAP * size[1]
    hh = size[1] / 2 + LEADER_GAP * size[1]
    dx, dy = tx - cx, ty - cy
    f = 1.0 / max(abs(dx) / hw, abs(dy) / hh, 1e-9)
    if f >= 1.0:
        return None
    sx, sy = cx + dx * f, cy + dy * f
    if np.hypot(tx - sx, ty - sy) < LEADER_MIN * size[1]:
        return None
    return ax.add_line(plt.Line2D([sx, tx], [sy, ty], color=colour, lw=1.3,
                                  zorder=19, solid_capstyle="round",
                                  path_effects=[patheffects.withStroke(
                                      linewidth=3.2, foreground=SURFACE)]))


def fit_ellipses(df, n_std=1.0, min_rows=20):
    """One ellipse per pitch type with enough pitches, busiest type first."""
    counts = df.pitch_type.value_counts()
    ells, order, skipped = {}, [], []
    for p in counts.index:
        e = group_ellipse(df.loc[df.pitch_type == p, "HRA"],
                          df.loc[df.pitch_type == p, "VRA"], n_std)
        if e is None or counts[p] < min_rows:
            skipped.append(f"{NAMES.get(p, p)} ({counts[p]})")
        else:
            ells[p] = e
            order.append(p)
    if not order:
        raise SystemExit("no pitch type had enough tracked pitches")
    return ells, order, skipped


def cluster_box(ells):
    """The bounding box of a set of ellipses, as (x0, x1, y0, y1)."""
    b = np.array([e.bounds for e in ells.values()])
    return b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max()


def square_frame(cx0, cx1, cy0, cy1):
    """The drawn frame around a cluster: its box, a margin, and squared off.

    The margin is where the pitch names live, so it has to be wide enough for
    one; the drawn axis is what a reader measures off, so it covers the same
    number of degrees each way and the short side grows out to the long one.
    """
    padx, pady = FRAME_PAD[0] * (cx1 - cx0), FRAME_PAD[1] * (cy1 - cy0)
    x0, x1, y0, y1 = cx0 - padx, cx1 + padx, cy0 - pady, cy1 + pady
    side = max(x1 - x0, y1 - y0)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    return xc - side / 2, xc + side / 2, yc - side / 2, yc + side / 2


def shared_frame(pitcher, season, spans, n_std=1.0, game_type="R",
                 min_rows=20):
    """One frame that holds every segment, so the charts can be read together.

    Each segment's cluster is fitted the way its own chart would fit it, and
    the frame is drawn round all of them at once - same centre, same span, so
    an ellipse that moved between segments is seen to have moved rather than
    just redrawn against a different ruler.
    """
    boxes = [cluster_box(fit_ellipses(
        load(pitcher, season, game_type, a, b)[1], n_std, min_rows)[0])
        for a, b in spans]
    return square_frame(min(c[0] for c in boxes), max(c[1] for c in boxes),
                        min(c[2] for c in boxes), max(c[3] for c in boxes))


def build(pitcher, season, n_std=1.0, game_type="R", min_rows=20,
          min_seg_area=0.10, start=None, end=None, limits=None):
    slug, df = load(pitcher, season, game_type, start, end)
    X, Y = "HRA", "VRA"
    year = int(df.game_date.dt.year.mode()[0])
    when = f"{year} {span_text(start, end)}".strip()

    ells, order, skipped = fit_ellipses(df, n_std, min_rows)

    hues = {p: HUES.get(p, HUES["UN"]) for p in order}
    summary = (df[df.pitch_type.isin(order)]
               .groupby("pitch_type", observed=True)
               .agg(n=("release_speed", "size"), velo=("release_speed", "mean"))
               .reindex(order))
    summary["pct"] = summary.n / summary.n.sum() * 100

    # ---------------------------------------------------------- limits
    # a frame handed in wins, so segments of one season can share one ruler
    x0, x1, y0, y1 = limits or square_frame(*cluster_box(ells))
    XLIM, YLIM, dw, dh = window(x0, x1, y0, y1)

    def fx(f):
        return XLIM[0] + f * dw

    def fy(f):
        return YLIM[0] + f * dh


    mpl.rcParams.update({
        "font.family": use_font(),
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    ratio = dw / dh
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI_PNG)
    ax = fig.add_axes([AX_L, B, AX_W, AX_H])
    # fixed here rather than at the end: placing the names measures them
    # through transData, which only means anything once the window is set
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ---------------------------------------------------------- axes
    # a frame on the cluster's own bounding box, with a degree grid beneath the
    # fills - it is a background reference, not an overlay
    xt, yt = nice_ticks(x0, x1), nice_ticks(y0, y1)
    for v in xt:
        ax.plot([v, v], [y0, y1], color=GRIDC, lw=0.8, zorder=0)
    for v in yt:
        ax.plot([x0, x1], [v, v], color=GRIDC, lw=0.8, zorder=0)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RULE,
                           lw=1.0, zorder=22))
    for v in xt:
        ax.text(v, y0 - 0.018 * dh, f"{v:g}", color=INK2, fontsize=14,
                ha="center", va="top")
    for v in yt:
        ax.text(x0 - 0.008 * dw, v, f"{v:g}", color=INK2, fontsize=14,
                ha="right", va="center")
    ax.text((x0 + x1) / 2, y0 - 0.060 * dh, "HRA (°)", color=INK2,
            fontsize=14, ha="center", va="top")
    ax.text(x0 - 0.048 * dw, (y0 + y1) / 2, "VRA (°)", color=INK2,
            fontsize=14, ha="center", va="center", rotation=90)

    # ---------------------------------------------------------- depth maps
    N = 1100
    gx, gy = np.meshgrid(np.linspace(*XLIM, N),
                         np.linspace(*YLIM, max(2, int(N / ratio))))
    masks = {p: ells[p].contains(gx, gy) for p in order}
    depth = sum(masks.values())
    share = sum(summary.loc[p, "pct"] / 100 * masks[p] for p in order)
    DMAX, SMAX = int(depth.max()), float(share.max())
    SMIN = float(share[depth >= 1].min())
    cell = (dw / gx.shape[1]) * (dh / gx.shape[0])

    # Segment map.  A segment is one exact set of overlapping ellipses - the
    # regions of a Venn diagram.  Each is shaded by how densely his pitches
    # actually landed in it, in pitches per square degree, which is empirical
    # concentration rather than the usage arithmetic the weighted map does.
    seg = np.zeros(depth.shape, dtype=np.int64)
    pseg = np.zeros(len(df), dtype=np.int64)
    pxs, pys = df[X].to_numpy(float), df[Y].to_numpy(float)
    for k, pt in enumerate(order):
        seg |= masks[pt].astype(np.int64) << k
        pseg |= ells[pt].contains(pxs, pys).astype(np.int64) << k
    hit = dict(zip(*np.unique(pseg, return_counts=True)))
    codes, inv, cells = np.unique(seg.ravel(), return_inverse=True,
                                  return_counts=True)
    seg_area = cells * cell
    seg_n = np.array([hit.get(int(c), 0) for c in codes], float)
    seg_dens = np.divide(seg_n, seg_area, out=np.zeros_like(seg_n),
                         where=seg_area > 0)
    dens = seg_dens[inv].reshape(seg.shape)
    inside = depth >= 1
    # A density is a ratio, so a sliver holding a handful of pitches returns an
    # enormous one.  Segments below min_seg_area are kept on the map but left
    # out of the scale, and clip into its top rather than stretching it.
    solid = (seg_area >= min_seg_area)[inv].reshape(seg.shape) & inside
    scaled = solid if solid.any() else inside
    KMIN, KMAX = float(dens[scaled].min()), float(dens[scaled].max())
    segments = sorted(((int(c), int(n), float(a), float(d),
                        bool(a >= min_seg_area))
                       for c, n, a, d in zip(codes, seg_n, seg_area, seg_dens)
                       if c != 0), key=lambda r: -r[3])

    cmap_c = LinearSegmentedColormap.from_list("share", lab_ramp(*RAMP_ENDS))
    steps = [cmap_c(v) for v in (np.linspace(0, 1, DMAX) if DMAX > 1
                                 else np.array([1.0]))]
    cmap_b = ListedColormap(steps)
    # min -> background, max -> white, same as the count scale: the shallowest
    # share present sits on the ramp's floor, not somewhere above it
    norm_c = Normalize(SMIN, SMAX)
    # One layer, blended in RGB, rather than two stacked translucent ones.
    # Stacking composites as wc*C + (1-wc)*(wb*B + (1-wb)*bg), so at the
    # half-way point a quarter of the background bleeds through and the map
    # visibly dims - a third look the transition passes through.  Mixing the
    # colours directly gives (1-mix)*B + mix*C, a straight hand-off.
    norm_b = BoundaryNorm(np.arange(0.5, DMAX + 1.5), cmap_b.N)
    rgb_b = cmap_b(norm_b(depth))[..., :3]
    rgb_c = cmap_c(norm_c(share))[..., :3]
    norm_d = Normalize(KMIN, KMAX)
    rgb_d = cmap_c(norm_d(dens))[..., :3]
    cover = (depth >= 1).astype(float)
    im = ax.imshow(np.zeros((*depth.shape, 4)), origin="lower",
                   extent=(*XLIM, *YLIM), interpolation="nearest", zorder=1)

    # ---------------------------------------------------------- ellipses
    edges, halos, names = [], [], {}
    for p in order:
        sub = df[df.pitch_type == p]
        x, y = sub[X].to_numpy(float), sub[Y].to_numpy(float)
        # the outline stays in the pitch colour throughout; the halo underneath
        # only fades in for the shaded states, where the fill goes dark
        halo = confidence_ellipse(x, y, ax, n_std, edgecolor=SURFACE, zorder=9)
        halo.set_linewidth(3.0)
        halos.append(halo)
        edges.append(confidence_ellipse(x, y, ax, n_std, edgecolor=hues[p],
                                        zorder=12))

        # parked on the ellipse for now - nothing can be placed until every
        # name's drawn size is known.  The dark stroke keeps it readable where
        # the shaded states run the fill up to white underneath it.
        r = summary.loc[p]
        names[p] = ax.text(ells[p].mean[0], ells[p].mean[1],
                           f"{NAMES.get(p, p)}  {r.pct:.1f}%",
                           color=hues[p], fontsize=LABEL_PT, va="center",
                           ha="center", fontweight="semibold", zorder=20,
                           path_effects=[patheffects.withStroke(
                               linewidth=4.0, foreground=SURFACE)])

    sizes = text_sizes(fig, ax, names)
    for p, (spot, anchor) in label_places(ells, order, sizes, (x0, x1, y0, y1),
                                          depth, XLIM, YLIM).items():
        names[p].set_position(spot)
        leader(ax, spot, anchor, sizes[p], hues[p])

    # ---------------------------------------------------------- legends
    # centred under the frame, on the same axis as the HRA label above it
    KH, KW = 0.035 * dh, 0.25 * dw
    KX, KY = (x0 + x1) / 2 - KW / 2, fy(0.07)
    key_b, key_c = [], []
    key_b.append(ax.text(KX + KW / 2, KY + KH + 0.018 * dh,
                         "Pitch types overlapping", color=INK, fontsize=16,
                         fontweight="semibold", va="bottom", ha="center"))
    gap = 0.002 * dw
    sw = (KW - gap * (DMAX - 1)) / DMAX
    for i, c in enumerate(steps):
        px = KX + i * (sw + gap)
        key_b.append(ax.add_patch(Rectangle((px, KY), sw, KH, facecolor=c,
                                            ec="none", zorder=2)))
        key_b.append(ax.text(px + sw / 2, KY - 0.013 * dh, str(i + 1),
                             color=INK2, fontsize=14, ha="center", va="top"))
    key_b.append(ax.add_patch(Rectangle((KX, KY), KW, KH, fill=False, ec=RULE,
                                        lw=0.7, zorder=3)))

    key_c.append(ax.text(KX + KW / 2, KY + KH + 0.018 * dh,
                         "Share of his pitches covering the spot", color=INK,
                         fontsize=16, fontweight="semibold", va="bottom",
                         ha="center"))
    key_c.append(ax.imshow(np.linspace(SMIN, SMAX, 256).reshape(1, -1),
                           cmap=cmap_c, norm=norm_c,
                           extent=(KX, KX + KW, KY, KY + KH),
                           interpolation="bilinear", zorder=2))
    key_c.append(ax.add_patch(Rectangle((KX, KY), KW, KH, fill=False, ec=RULE,
                                        lw=0.7, zorder=3)))
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        key_c.append(ax.text(KX + f * KW, KY - 0.013 * dh,
                             f"{(SMIN + f * (SMAX - SMIN)) * 100:.0f}%",
                             color=INK2, fontsize=14, ha="center", va="top"))

    key_d = []
    key_d.append(ax.text(KX + KW / 2, KY + KH + 0.018 * dh,
                         "Pitches per square degree", color=INK, fontsize=16,
                         fontweight="semibold", va="bottom", ha="center"))
    key_d.append(ax.imshow(np.linspace(KMIN, KMAX, 256).reshape(1, -1),
                           cmap=cmap_c, norm=norm_d,
                           extent=(KX, KX + KW, KY, KY + KH),
                           interpolation="bilinear", zorder=2))
    key_d.append(ax.add_patch(Rectangle((KX, KY), KW, KH, fill=False, ec=RULE,
                                        lw=0.7, zorder=3)))
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        key_d.append(ax.text(KX + f * KW, KY - 0.013 * dh,
                             f"{KMIN + f * (KMAX - KMIN):.0f}", color=INK2,
                             fontsize=14, ha="center", va="top"))

    # ---------------------------------------------------------- titles
    # centred in the header band, between the top of the figure and the rule
    # the segment rides the second line, after whatever that line already
    # says, so the headline itself reads the same whole season or not
    span = span_text(start, end)
    titles = [fig.text(L, (1.0 + RULE_Y) / 2,
                       f"{pitcher} {year} {head}\n"
                       + ", ".join(t for t in (line2, span) if t),
                       color=TITLE, fontsize=24, fontweight="semibold",
                       va="center") for head, line2 in (
        ("Release Angles", ""),
        ("Release Angle Overlap", "Number of Pitch Types"),
        ("Release Angle Overlap", "Weighted by Usage"),
        ("Release Angle Overlap", "Pitch Concentration"))]
    fig.lines.append(plt.Line2D([L, L + W_FRAC], [RULE_Y, RULE_Y],
                                transform=fig.transFigure, color=RULE, lw=0.8))
    # the footer is one block of two lines on the left, with the word mark
    # opposite it on the right; both sit on the same baseline band
    fig.text(L, FOOT_Y, "Angles as the ball leaves the hand\n"
             "Data: MLB StatsAPI", color=INK3, fontsize=12, va="bottom",
             linespacing=1.6)

    mark = watermark()
    if mark is not None:
        # its own axes, in figure fractions, so it keeps its size and aspect
        # whatever dpi the still or the gif is written at
        h = WATERMARK_W * FIG_W / FIG_H / (mark.shape[1] / mark.shape[0])
        wax = fig.add_axes([L + W_FRAC - WATERMARK_W, FOOT_Y, WATERMARK_W, h],
                           zorder=5)
        wax.imshow(mark, interpolation="antialiased")
        wax.axis("off")
        wax.patch.set_alpha(0)

    def weights(frame):
        """(wA, wB, wC) for a frame of the hold / cross-fade / hold cycle."""
        s, k = divmod(frame, HOLD + TRANS)
        w = [0.0] * STATES
        if k < HOLD:
            w[s] = 1.0
        else:
            p = (k - HOLD + 1) / TRANS
            u = p * p * (3 - 2 * p)          # smoothstep, so the ends settle
            w[s], w[(s + 1) % STATES] = 1 - u, u
        return w

    def swap_weights(frame):
        """Staggered, so title and legend never overlap mid-fade."""
        s, k = divmod(frame, HOLD + TRANS)
        w = [0.0] * STATES
        if k < HOLD:
            w[s] = 1.0
        else:
            p = (k - HOLD + 1) / TRANS
            w[s] = max(0.0, 1 - 2 * p)
            w[(s + 1) % STATES] = max(0.0, 2 * p - 1)
        return w

    def set_alpha(artists, a):
        for art in artists:
            art.set_alpha(None if a >= 1 else a)

    def paint(mw, tw):
        """mw: map weights over (outline, count, share, segment); tw: the same
        for the title and legend, which swap staggered rather than blended."""
        wa, wb, wc, wd = mw
        shaded = wb + wc + wd
        im.set_visible(shaded > 0)
        if shaded > 0:
            rgba = np.empty((*cover.shape, 4))
            rgba[..., :3] = (wb * rgb_b + wc * rgb_c + wd * rgb_d) / shaded
            rgba[..., 3] = cover * shaded
            im.set_data(rgba)
        set_alpha(halos, shaded)
        # outlines never change colour, only the weight each still image uses
        for e in edges:
            e.set_linewidth(2.0 * wa + 1.4 * shaded)
        for t, w in zip(titles, tw):
            t.set_alpha(w)
        for keys, w in zip((key_b, key_c, key_d), tw[1:]):
            set_alpha(keys, w)

    def update(frame):
        """One frame of the four-state loop."""
        paint(weights(frame), swap_weights(frame))
        return []

    def still(i):
        w = [0.0] * STATES
        w[i] = 1.0
        paint(w, w)

    return dict(fig=fig, update=update, still=still, slug=slug, year=year,
                tag=span_tag(start, end), when=when, games=df.game_date.nunique(),
                df=df, order=order, summary=summary, depth=depth, share=share,
                masks=masks, cell=cell, DMAX=DMAX, SMAX=SMAX, ells=ells,
                skipped=skipped, n_std=n_std, segments=segments,
                total=len(df), KMIN=KMIN, KMAX=KMAX,
                min_seg_area=min_seg_area)


def report(b):
    """The overlap-depth table, printed."""
    df, order, summary = b["df"], b["order"], b["summary"]
    depth, share, masks, cell = b["depth"], b["share"], b["masks"], b["cell"]
    w = {p: summary.loc[p, "pct"] / 100 for p in order}
    rows = []
    for p in order:
        m = masks[p]
        rows.append((NAMES.get(p, p), summary.loc[p, "n"], summary.loc[p, "pct"],
                     b["ells"][p].area, depth[m].mean(), share[m].mean() * 100,
                     (share[m].mean() - w[p]) * 100))
    t = pd.DataFrame(rows, columns=["pitch", "n", "usage%", "area_sqdeg",
                                    "mean_depth", "mean_share%", "excl_self%"])
    print(t.round(2).to_string(index=False))
    cov = depth >= 1
    print(f"\nusage-weighted mean_depth : "
          f"{sum(w[p] * depth[masks[p]].mean() for p in order):.3f}")
    print(f"union mean depth          : {depth[cov].mean():.3f}")
    print(f"union area                : {cov.sum() * cell:.2f} sq deg")
    print(f"max depth                 : {b['DMAX']}")
    print(f"peak weighted share       : {b['SMAX'] * 100:.1f}%")
    sel = share >= b["SMAX"] - 1e-9
    inside = [NAMES.get(p, p) for p in order
              if (masks[p] & sel).sum() / sel.sum() > 0.99]
    print(f"  peak is {len(inside)} types: {', '.join(inside)}")
    print(f"\ntop segments, by pitch concentration:")
    for code, n, area, d, ok in b["segments"][:8]:
        members = [NAMES.get(q, q) for k, q in enumerate(order)
                   if code >> k & 1]
        print(f"  {d:6.1f} /sq deg  {n:5,d} pitches in {area:5.2f} sq deg  "
              f"{'  ' if ok else '* '}{' + '.join(members)}")
    clipped = [r for r in b["segments"] if not r[4]]
    if clipped:
        print(f"  * {len(clipped)} segment(s) under {b['min_seg_area']:.2f} "
              f"sq deg, left out of the scale and clipped into its top: "
              f"{sum(r[2] for r in clipped):.2f} sq deg, "
              f"{sum(r[1] for r in clipped):,} pitches")
    if b["skipped"]:
        print("skipped (too few pitches): " + ", ".join(b["skipped"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pitcher", help='name ("Roki Sasaki") or MLBAM id')
    ap.add_argument("season", type=int)
    ap.add_argument("-t", "--game-type", default="R")
    ap.add_argument("-n", "--n-std", type=float, default=1.0)
    ap.add_argument("-m", "--min-rows", type=int, default=20,
                    help="drop pitch types with fewer pitches than this; 3 is "
                         "the floor a covariance ellipse needs, but a handful "
                         "of pitches makes a huge, meaningless one, so the "
                         "default asks for a real sample")
    ap.add_argument("-a", "--min-segment-area", type=float, default=0.10,
                    help="segments smaller than this (sq deg) stay on the "
                         "concentration map but are left out of its scale; "
                         "0 to use every segment")
    ap.add_argument("--start", help="first game date of the segment, "
                                    "\"7/31\" or \"2026-07-31\"")
    ap.add_argument("--end", help="last game date of the segment")
    ap.add_argument("--segment", action="append", metavar="START..END",
                    help="a date segment, either end open: \"..6/17\", "
                         "\"7/31..\", \"4/1..5/15\".  Repeat it, and every "
                         "segment is drawn on one shared frame, so the charts "
                         "can be read against each other")
    ap.add_argument("--stills", action="store_true", help="skip the gif")
    a = ap.parse_args(argv)

    if a.segment:
        spans = [tuple(None if not t.strip() else as_date(t, a.season)
                       for t in seg.split("..", 1))
                 for seg in a.segment]
    else:
        spans = [(None if a.start is None else as_date(a.start, a.season),
                  None if a.end is None else as_date(a.end, a.season))]
    # one segment sizes its own frame; several share one, or they would each
    # be drawn against a different ruler
    limits = (shared_frame(a.pitcher, a.season, spans, a.n_std, a.game_type,
                           a.min_rows) if len(spans) > 1 else None)
    for start, end in spans:
        draw(build(a.pitcher, a.season, a.n_std, a.game_type, a.min_rows,
                   a.min_segment_area, start, end, limits), a.stills)


def draw(b, stills):
    stem = f"{b['slug']}_{b['year']}{b['tag']}_angles"
    for i, tag in enumerate(("type", "count", "share", "segment")):
        b["still"](i)
        b["fig"].savefig(f"{stem}_{tag}.png", dpi=DPI_PNG)
        print(f"wrote {stem}_{tag}.png")
    if not stills:
        frames = STATES * (HOLD + TRANS)
        anim = FuncAnimation(b["fig"], b["update"], frames=frames,
                             interval=1000 / FPS)
        anim.save(f"{stem}.gif", writer=PillowWriter(fps=FPS), dpi=DPI_GIF)
        print(f"wrote {stem}.gif  ({frames} frames @ {FPS} fps)")
    plt.close(b["fig"])
    print()
    report(b)


if __name__ == "__main__":
    main()
