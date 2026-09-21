"""Color science for Closet OS: raw sensor counts -> sRGB -> XYZ -> CIELAB -> LCh.

Pure standard library. No numpy, no external color library -- the conversion
chain is hand-rolled here so the whole thing is inspectable and unit-testable
in one file.

Scope note: this module covers the measurement path end to end, from the
TCS34725's raw counts through to LCh. Color *difference* (CIEDE2000) is
deliberately not here -- it is built and validated as its own step, against
the Sharma et al. test vectors retained in tests/sharma_ciede2000.py.

Conventions
-----------
* sRGB components are 0-255 (integers or floats), IEC 61966-2-1 transfer curve.
* XYZ is scaled 0-100, referenced to D65 / 2-degree observer.
* Lab is CIE 1976 L*a*b*.
* Hue angles are degrees in [0, 360).
"""

from __future__ import annotations

import math
from typing import NamedTuple

__all__ = [
    "Calibration",
    "Lab",
    "LCh",
    "XYZ",
    "D65_WHITE",
    "SRGB_MAX",
    "normalize",
    "hex_to_srgb",
    "srgb_to_linear",
    "linear_to_srgb",
    "srgb_to_xyz",
    "xyz_to_lab",
    "srgb_to_lab",
    "rgb_to_lab",
    "hex_to_lab",
    "lab_to_lch",
    "lch_to_lab",
]


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

class XYZ(NamedTuple):
    x: float
    y: float
    z: float


class Lab(NamedTuple):
    l: float
    a: float
    b: float


class LCh(NamedTuple):
    l: float
    c: float
    h: float


class Calibration(NamedTuple):
    """Per-channel white-balance factors from a white-reference reading.

    Each factor is the Clear-normalised ratio the sensor reports for a known
    neutral card, so dividing a measurement's ratio by it maps that card back
    onto equal R=G=B. Factors default to 1.0, the identity transform -- useful
    for tests and for a station that has not been calibrated yet.

    Persisting these is not implemented; that lands with the calibration table.
    """

    wr: float = 1.0
    wg: float = 1.0
    wb: float = 1.0


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: D65 reference white, 2-degree observer, Y scaled to 100.
D65_WHITE = XYZ(95.047, 100.000, 108.883)

# Linear-sRGB -> XYZ (D65). These are the 7-decimal coefficients whose rows sum
# *exactly* to D65_WHITE above (0.95047 / 1.00000 / 1.08883). That consistency
# is what makes pure white land on L*=100, a*=b*=0 instead of a*=0.005ish; a
# 4-decimal matrix leaves a visible residual on the white patch.
#
# Rows are X, Y, Z. Transposing this matrix is the exact failure mode called
# out in the handoff -- test_color.py pins all six reference swatches, and a
# transpose moves every one of them.
_SRGB_TO_XYZ_PUBLISHED = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def _normalised_to_white(matrix, white):
    """Rescale each row so it sums *exactly* to its white-point component.

    The published coefficients are quoted to 7 decimals, so the Y row sums to
    1.0000001 rather than 1. Left alone that puts pure white at L* = 100.000004
    and gives every neutral grey an a* of about -1.7e-5 instead of hard zero.

    The error is perceptually nil, but "r == g == b implies a* == b* == 0" is
    worth having as an exact structural guarantee rather than an approximate
    one: scoring.py keys its neutral-wildcard branch off C* < 12, and
    identify.py ranks garments by chroma-sensitive distance, so the neutral
    axis is load-bearing. The rescale is a relative adjustment of ~1e-7 per
    coefficient and moves no reference swatch at 4 decimal places.
    """
    targets = (white.x / 100.0, white.y / 100.0, white.z / 100.0)
    return tuple(
        tuple(coeff * target / sum(row) for coeff in row)
        for row, target in zip(matrix, targets)
    )


#: Linear-sRGB -> XYZ (D65), white-point normalised. See above.
SRGB_TO_XYZ = _normalised_to_white(_SRGB_TO_XYZ_PUBLISHED, D65_WHITE)

# CIE standard constants, as exact rationals rather than the rounded 0.008856 /
# 903.3 that float around. Using the rationals keeps the two branches of f(t)
# continuous at the join.
CIE_EPSILON = 216.0 / 24389.0     # 0.008856451679...
CIE_KAPPA = 24389.0 / 27.0        # 903.296296296...

# sRGB transfer curve (IEC 61966-2-1).
_SRGB_LINEAR_CUTOFF = 0.04045
_SRGB_GAMMA_CUTOFF = 0.0031308
_SRGB_SLOPE = 12.92
_SRGB_ALPHA = 0.055
_SRGB_GAMMA = 2.4

#: Full-scale sRGB component.
SRGB_MAX = 255.0


# --------------------------------------------------------------------------
# Raw sensor counts -> sRGB
# --------------------------------------------------------------------------

def normalize(r: float, g: float, b: float, c: float, cal: Calibration):
    """Convert raw TCS34725 counts to 0-255 sRGB.

    The chain, in order:

    1. Divide each of ``r``, ``g``, ``b`` by the Clear channel ``c``. This is
       what makes the reading independent of exposure and ambient level -- two
       measurements of the same garment under different light produce the same
       ratios even though every raw count changed.
    2. Divide each ratio by its calibration factor (``cal.wr`` / ``cal.wg`` /
       ``cal.wb``), which pulls the sensor's spectral response and the LED's
       tint back onto neutral.
    3. Scale to 0-255 and clamp to that range.

    Steps 1 and 2 are both divisions, so arithmetically they commute; the order
    is stated because step 3's clamp does *not* commute with them, and because
    a calibration factor only means anything as a ratio against Clear.

    ``c == 0`` means the sensor saw no light at all -- a dead LED, a covered
    aperture, or an integration cycle that never ran. There is no reading to
    salvage, so this raises before any arithmetic happens rather than letting
    a ZeroDivisionError surface from the middle of the chain.

    Returns floats, not ints. Rounding is the caller's decision: the Lab
    conversion downstream is happier with the full precision, and quantising
    here would throw away resolution the sensor actually has.

    :raises ValueError: if ``c`` is zero or negative, or if any calibration
        factor is zero or negative.
    """
    # Guard first -- before reading `cal`, before any division. The tests pin
    # this ordering, because "ValueError" and "ZeroDivisionError from three
    # lines deeper" are very different things to debug from a station log.
    if c <= 0:
        raise ValueError(
            f"clear channel must be positive, got {c!r}; "
            "a zero Clear reading means no light reached the sensor"
        )
    if cal.wr <= 0 or cal.wg <= 0 or cal.wb <= 0:
        raise ValueError(f"calibration factors must be positive, got {cal!r}")

    return tuple(
        _clamp_srgb(SRGB_MAX * (raw / c) / factor)
        for raw, factor in ((r, cal.wr), (g, cal.wg), (b, cal.wb))
    )


def _clamp_srgb(value: float) -> float:
    """Clamp to the representable sRGB range."""
    if value < 0.0:
        return 0.0
    if value > SRGB_MAX:
        return SRGB_MAX
    return value


# --------------------------------------------------------------------------
# sRGB -> XYZ -> Lab
# --------------------------------------------------------------------------

def hex_to_srgb(value: str) -> tuple[int, int, int]:
    """Parse ``"#RRGGBB"`` (or ``"RGB"``) into an 0-255 sRGB triple."""
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"not a 6-digit hex color: {value!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def srgb_to_linear(channel: float) -> float:
    """Undo the sRGB gamma curve. Input and output are 0-1."""
    if channel <= _SRGB_LINEAR_CUTOFF:
        return channel / _SRGB_SLOPE
    return ((channel + _SRGB_ALPHA) / (1.0 + _SRGB_ALPHA)) ** _SRGB_GAMMA


def linear_to_srgb(channel: float) -> float:
    """Apply the sRGB gamma curve. Input and output are 0-1."""
    if channel <= _SRGB_GAMMA_CUTOFF:
        return channel * _SRGB_SLOPE
    return (1.0 + _SRGB_ALPHA) * (channel ** (1.0 / _SRGB_GAMMA)) - _SRGB_ALPHA


def srgb_to_xyz(r: float, g: float, b: float) -> XYZ:
    """Convert 0-255 sRGB to D65 XYZ scaled 0-100."""
    rl = srgb_to_linear(r / 255.0)
    gl = srgb_to_linear(g / 255.0)
    bl = srgb_to_linear(b / 255.0)
    (mx, my, mz) = SRGB_TO_XYZ
    return XYZ(
        100.0 * (mx[0] * rl + mx[1] * gl + mx[2] * bl),
        100.0 * (my[0] * rl + my[1] * gl + my[2] * bl),
        100.0 * (mz[0] * rl + mz[1] * gl + mz[2] * bl),
    )


def _f(t: float) -> float:
    """CIELAB nonlinearity."""
    if t > CIE_EPSILON:
        return t ** (1.0 / 3.0)
    return (CIE_KAPPA * t + 16.0) / 116.0


def xyz_to_lab(x: float, y: float, z: float, white: XYZ = D65_WHITE) -> Lab:
    """Convert XYZ (0-100) to CIE L*a*b*."""
    fx = _f(x / white.x)
    fy = _f(y / white.y)
    fz = _f(z / white.z)
    return Lab(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def srgb_to_lab(r: float, g: float, b: float, white: XYZ = D65_WHITE) -> Lab:
    """Convert 0-255 sRGB straight through to CIE L*a*b*.

    Three stages: the sRGB gamma linearisation (IEC 61966-2-1), the
    linear-sRGB -> XYZ matrix, and the XYZ -> Lab nonlinear compression.

    ``white`` defaults to D65 (95.047, 100.000, 108.883). The matrix in stage
    two is itself D65-referenced, so passing a different white here adapts
    only the final compression -- it is the right knob for comparing against a
    converter that quotes a different D65 rounding, not for a genuine change
    of illuminant. Note also that it cannot move L*: that is
    ``116*f(Y/Yn) - 16`` and every D65 in circulation puts Yn at exactly
    100.000, so only a* and b* respond.
    """
    return xyz_to_lab(*srgb_to_xyz(r, g, b), white=white)


#: Spelling of :func:`srgb_to_lab` used by the measurement path, where the
#: input has come from :func:`normalize` rather than from a hex literal.
rgb_to_lab = srgb_to_lab


def hex_to_lab(value: str) -> Lab:
    """Convert ``"#RRGGBB"`` straight through to CIE L*a*b*."""
    return srgb_to_lab(*hex_to_srgb(value))


# --------------------------------------------------------------------------
# Lab <-> LCh
# --------------------------------------------------------------------------

def lab_to_lch(l: float, a: float, b: float) -> LCh:
    """Convert L*a*b* to cylindrical L*C*h.

    ``C = hypot(a, b)`` and ``h = atan2(b, a)`` wrapped into [0, 360).
    A fully neutral color (a == b == 0) has an undefined hue; we report 0.0.
    """
    c = math.hypot(a, b)
    if a == 0.0 and b == 0.0:
        return LCh(l, 0.0, 0.0)
    return LCh(l, c, math.degrees(math.atan2(b, a)) % 360.0)


def lch_to_lab(l: float, c: float, h: float) -> Lab:
    """Inverse of :func:`lab_to_lch`."""
    rad = math.radians(h)
    return Lab(l, c * math.cos(rad), c * math.sin(rad))
