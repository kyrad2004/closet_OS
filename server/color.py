"""Color science for Closet OS: sRGB -> XYZ -> CIELAB -> LCh, and CIEDE2000.

Pure standard library. No numpy, no external color library -- the conversion
chain and the delta-E formula are hand-rolled here so the whole thing is
inspectable and unit-testable in one file.

Scope note: this module currently covers the *colorimetric* half of the chain
(sRGB -> Lab -> LCh -> dE2000). Sensor-side steps -- normalising raw TCS34725
counts by the Clear channel and applying the user's stored white-balance
factors -- are deliberately not here yet; they land with the calibration table.

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
    "Lab",
    "LCh",
    "XYZ",
    "D65_WHITE",
    "hex_to_srgb",
    "srgb_to_linear",
    "linear_to_srgb",
    "srgb_to_xyz",
    "xyz_to_lab",
    "srgb_to_lab",
    "hex_to_lab",
    "lab_to_lch",
    "lch_to_lab",
    "delta_e_2000",
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

# 25**7, used twice in CIEDE2000.
_POW_25_7 = 25.0 ** 7

# Tolerance on the |h1' - h2'| == 180 branch boundary in CIEDE2000.
#
# When two hues are exactly antipodal the mean hue is genuinely ambiguous and
# the formula's branch test sits precisely on the knife edge, so whether
# atan2 returns 180.0 or 180.00000000000003 decides the answer. Sharma pairs
# 10 and 14 are exactly there (measured margins -2.8e-14 and 0.0), and the
# published values correspond to the "<= 180" branch. Without a tolerance the
# result depends on the platform's libm: numpy takes the other branch on
# Linux, which is why colour-science excludes pair 14 from its test suite.
#
# 1e-10 is ~4 orders of magnitude above the float noise at the boundary and
# ~7 below the nearest genuinely-over-180 case in the dataset (pair 11, at
# +1.5e-3), so it disambiguates the degenerate case without reclassifying any
# real one.
_HUE_BOUNDARY_TOL = 1e-10


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


def srgb_to_lab(r: float, g: float, b: float) -> Lab:
    """Convert 0-255 sRGB straight through to CIE L*a*b*."""
    return xyz_to_lab(*srgb_to_xyz(r, g, b))


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


# --------------------------------------------------------------------------
# CIEDE2000
# --------------------------------------------------------------------------

def _hue_prime(a_prime: float, b: float) -> float:
    """Hue angle in [0, 360) for the a'-adjusted coordinates.

    Per Sharma et al., the angle is defined as 0 when both components are 0
    (``atan2(0, 0)`` is 0 in IEEE terms anyway, but being explicit documents
    that this is a deliberate convention and not an accident).
    """
    if a_prime == 0.0 and b == 0.0:
        return 0.0
    return math.degrees(math.atan2(b, a_prime)) % 360.0


def delta_e_2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
    k_l: float = 1.0,
    k_c: float = 1.0,
    k_h: float = 1.0,
) -> float:
    """CIEDE2000 color difference between two CIELAB colors.

    Implemented from Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference
    formula: Implementation notes, supplementary test data, and mathematical
    observations", Color Research & Application 30(1), 21-30.

    The four traps that paper calls out are handled explicitly and each is
    covered by the Sharma test vectors in tests/test_color.py:

    1. ``dh'`` quadrant selection -- the +/-360 wrap below.
    2. ``hbar'`` when the two hues straddle 0/360 -- the ``h_sum < 360`` branch.
    3. ``C1' * C2' == 0`` degenerate cases -- ``dh'`` forced to 0 and ``hbar'``
       taken as the plain sum, so a neutral never injects a bogus hue term.
    4. The sign of ``R_T`` -- it is negative, and it multiplies the *scaled*
       chroma and hue terms, not the raw deltas.

    A fifth trap the paper implies but does not spell out: the ``<= 180``
    branch tests are exactly on a knife edge for antipodal hues. See
    ``_HUE_BOUNDARY_TOL``.
    """
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    # --- Step 1: chroma, and the a* expansion that pulls near-neutrals apart --
    c1_ab = math.hypot(a1, b1)
    c2_ab = math.hypot(a2, b2)
    c_bar_ab_7 = (0.5 * (c1_ab + c2_ab)) ** 7
    g = 0.5 * (1.0 - math.sqrt(c_bar_ab_7 / (c_bar_ab_7 + _POW_25_7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2

    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)

    h1p = _hue_prime(a1p, b1)
    h2p = _hue_prime(a2p, b2)

    # --- Step 2: the deltas ------------------------------------------------
    delta_lp = l2 - l1
    delta_cp = c2p - c1p

    c_product = c1p * c2p

    # Trap 1 + trap 3: pick the dh' representative in (-180, 180]; if either
    # color is neutral there is no meaningful hue difference at all.
    if c_product == 0.0:
        delta_hp = 0.0
    else:
        diff = h2p - h1p
        if abs(diff) <= 180.0 + _HUE_BOUNDARY_TOL:
            delta_hp = diff
        elif diff > 180.0:
            delta_hp = diff - 360.0
        else:
            delta_hp = diff + 360.0

    delta_cap_hp = 2.0 * math.sqrt(c_product) * math.sin(math.radians(delta_hp) / 2.0)

    # --- Step 3: the weighting functions -----------------------------------
    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    # Trap 2 + trap 3: mean hue has to cross the 0/360 seam the short way.
    if c_product == 0.0:
        h_bar_p = h1p + h2p
    else:
        h_sum = h1p + h2p
        if abs(h1p - h2p) <= 180.0 + _HUE_BOUNDARY_TOL:
            h_bar_p = h_sum / 2.0
        elif h_sum < 360.0:
            h_bar_p = (h_sum + 360.0) / 2.0
        else:
            h_bar_p = (h_sum - 360.0) / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    delta_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p_7 = c_bar_p ** 7
    r_c = 2.0 * math.sqrt(c_bar_p_7 / (c_bar_p_7 + _POW_25_7))

    # Trap 4: R_T is negative.
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c

    l_offset_sq = (l_bar_p - 50.0) ** 2
    s_l = 1.0 + (0.015 * l_offset_sq) / math.sqrt(20.0 + l_offset_sq)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t

    # --- Step 4: combine ----------------------------------------------------
    term_l = delta_lp / (k_l * s_l)
    term_c = delta_cp / (k_c * s_c)
    term_h = delta_cap_hp / (k_h * s_h)

    return math.sqrt(
        term_l * term_l
        + term_c * term_c
        + term_h * term_h
        + r_t * term_c * term_h
    )
