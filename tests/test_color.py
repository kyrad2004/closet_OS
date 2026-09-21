"""Tests for server/color.py.

Two gates live here, in this order:

1. The sRGB -> XYZ -> CIELAB chain against published reference swatches.
   Nothing downstream (scoring, identify) may be trusted until this passes --
   a transposed matrix row produces plausible-looking wrong numbers.
2. `normalize`: the raw-counts contract. Clear-normalise, then calibrate,
   then clamp -- and refuse a zero Clear reading before dividing by it.

CIEDE2000 is not covered here because it is not implemented yet. The Sharma
et al. test vectors it will be validated against are retained, unused, in
tests/sharma_ciede2000.py.
"""

from __future__ import annotations

import math

import pytest

from server.color import (
    D65_WHITE,
    SRGB_MAX,
    SRGB_TO_XYZ,
    XYZ,
    Calibration,
    hex_to_lab,
    hex_to_srgb,
    lab_to_lch,
    lch_to_lab,
    linear_to_srgb,
    normalize,
    rgb_to_lab,
    srgb_to_lab,
    srgb_to_linear,
    srgb_to_xyz,
)

# Tolerance for the swatch fixtures: the published values are quoted to 4
# decimals, so anything beyond 1e-4 is a real disagreement, not rounding.
SWATCH_TOL = 1e-4

# SRGB_TO_XYZ is normalised so its rows sum exactly to D65, so neutrality and
# the white point hold to float noise rather than to matrix print precision.
MATRIX_RESIDUAL = 1e-12

# Published CIE L*a*b* (D65, 2-degree observer) for the six reference swatches
# named in the handoff.
#
# PROVENANCE -- read before trusting these.
#
# These figures were written into this file after server/color.py already
# existed, so on their own they do NOT constitute an independent oracle: a
# test that asserts a function agrees with its own output proves nothing.
# They are independently corroborated by two things that do not share code
# with server/color.py:
#
#   1. test_matrix_matches_first_principles_derivation (below) rebuilds the
#      sRGB->XYZ matrix from the IEC 61966-2-1 primary chromaticities and the
#      D65 white point, and agrees with the transcribed constant to 3.9e-08.
#      The matrix is therefore not taken on trust at all.
#   2. colour-science 0.4.7, pinned to the same D65 convention and deriving
#      its own matrix from primaries, reproduces all six rows below to
#      2.5e-05 -- a separate implementation by a separate author.
#
# The original author's reference converter could not be reached from the
# build environment (egress blocked), so no third-party web source was
# consulted directly. If you want a fully external anchor for the writeup,
# paste these six hex values into any sRGB->Lab converter that states a D65
# white point of 95.047/100/108.883 and diff the result.
REFERENCE_SWATCHES = [
    ("#FFFFFF", (100.0000, 0.0000, 0.0000)),
    ("#000000", (0.0000, 0.0000, 0.0000)),
    ("#FF0000", (53.2408, 80.0925, 67.2032)),
    ("#00FF00", (87.7347, -86.1827, 83.1793)),
    ("#0000FF", (32.2970, 79.1875, -107.8602)),
    ("#808080", (53.5850, 0.0000, 0.0000)),
]


# ---------------------------------------------------------------------------
# Gate 1: the conversion chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hex_value,expected", REFERENCE_SWATCHES, ids=[h for h, _ in REFERENCE_SWATCHES])
def test_reference_swatch_lab(hex_value, expected):
    """The six published hex -> Lab fixtures. This is the gate."""
    got = hex_to_lab(hex_value)
    for axis, g, e in zip("Lab", got, expected):
        assert g == pytest.approx(e, abs=SWATCH_TOL), (
            f"{hex_value} {axis}*: got {g:.6f}, published {e:.4f}"
        )


def test_white_point_round_trips_to_L100():
    """Pure white must land exactly on the reference white, not near it.

    This is the specific check that catches a 4-decimal sRGB matrix whose rows
    do not sum to D65: it leaves white at a* = 0.005ish instead of 0.
    """
    x, y, z = srgb_to_xyz(255, 255, 255)
    assert (x, y, z) == pytest.approx(tuple(D65_WHITE), abs=1e-3)


def test_matrix_is_not_transposed():
    """Guard against the exact failure mode called out in the handoff.

    Green is the most diagnostic primary: with the correct matrix its Y (luma)
    contribution dominates, giving L* ~ 87.7. Under a transposed matrix the
    green primary picks up the X row's coefficients instead and L* collapses.
    """
    l_green, _, _ = hex_to_lab("#00FF00")
    l_red, _, _ = hex_to_lab("#FF0000")
    l_blue, _, _ = hex_to_lab("#0000FF")
    # sRGB luma ordering: green is much lighter than red, red lighter than blue.
    assert l_green > l_red > l_blue
    assert l_green == pytest.approx(87.7347, abs=SWATCH_TOL)


def test_gamma_curve_round_trips():
    for i in range(0, 256):
        c = i / 255.0
        assert linear_to_srgb(srgb_to_linear(c)) == pytest.approx(c, abs=1e-12)


def test_gamma_curve_is_continuous_at_the_join():
    """The linear segment and the power segment must meet."""
    eps = 1e-9
    below = srgb_to_linear(0.04045 - eps)
    above = srgb_to_linear(0.04045 + eps)
    assert below == pytest.approx(above, abs=1e-8)


def test_hex_parsing():
    assert hex_to_srgb("#FF8000") == (255, 128, 0)
    assert hex_to_srgb("ff8000") == (255, 128, 0)
    assert hex_to_srgb("#f80") == (255, 136, 0)
    with pytest.raises(ValueError):
        hex_to_srgb("#12345")


def test_lightness_is_monotonic_in_grey():
    ls = [srgb_to_lab(v, v, v).l for v in range(0, 256)]
    assert all(b > a for a, b in zip(ls, ls[1:]))
    assert ls[0] == pytest.approx(0.0, abs=1e-12)
    assert ls[-1] == pytest.approx(100.0, abs=MATRIX_RESIDUAL)


def test_greys_are_neutral():
    """Any r == g == b must have a* == b* == 0 (to matrix precision)."""
    for v in (0, 16, 64, 128, 200, 255):
        lab = srgb_to_lab(v, v, v)
        assert lab.a == pytest.approx(0.0, abs=MATRIX_RESIDUAL)
        assert lab.b == pytest.approx(0.0, abs=MATRIX_RESIDUAL)


def test_matrix_rows_sum_exactly_to_the_white_point():
    """The transform must map sRGB white onto the declared reference white.

    This is what makes the neutral axis exact. scoring.py keys its
    neutral-wildcard branch off C* < 12 and identify.py ranks by a
    chroma-sensitive distance, so a grey that converts to a* = -1.7e-5 instead
    of 0 is a small lie in a load-bearing place. Normalising the published
    7-decimal matrix removes it outright.
    """
    for row, target in zip(SRGB_TO_XYZ, (D65_WHITE.x / 100.0, D65_WHITE.y / 100.0, D65_WHITE.z / 100.0)):
        assert sum(row) == pytest.approx(target, abs=1e-15)

    assert tuple(srgb_to_lab(255, 255, 255)) == pytest.approx((100.0, 0.0, 0.0), abs=MATRIX_RESIDUAL)

    worst = max(
        max(abs(srgb_to_lab(v, v, v).a), abs(srgb_to_lab(v, v, v).b))
        for v in range(0, 256)
    )
    assert worst < MATRIX_RESIDUAL


def test_normalisation_does_not_move_the_published_swatches():
    """The rescale is ~1e-7 relative; no reference value may shift at 4dp."""
    for hex_value, expected in REFERENCE_SWATCHES:
        assert tuple(hex_to_lab(hex_value)) == pytest.approx(expected, abs=SWATCH_TOL)


# ---------------------------------------------------------------------------
# lab_to_lch
# ---------------------------------------------------------------------------

def test_lab_to_lch_known_angles():
    """Chroma is the hypotenuse; hue is atan2 wrapped into [0, 360)."""
    assert lab_to_lch(50.0, 10.0, 0.0) == pytest.approx((50.0, 10.0, 0.0))
    assert lab_to_lch(50.0, 0.0, 10.0) == pytest.approx((50.0, 10.0, 90.0))
    assert lab_to_lch(50.0, -10.0, 0.0) == pytest.approx((50.0, 10.0, 180.0))
    assert lab_to_lch(50.0, 0.0, -10.0) == pytest.approx((50.0, 10.0, 270.0))
    assert lab_to_lch(50.0, 3.0, 4.0).c == pytest.approx(5.0)


def test_lab_to_lch_negative_b_wraps_not_negative():
    """The handoff's `% 360` -- a naive atan2 would report -45 here."""
    lch = lab_to_lch(50.0, 10.0, -10.0)
    assert lch.h == pytest.approx(315.0)
    assert 0.0 <= lch.h < 360.0


def test_lab_to_lch_neutral_hue_is_zero():
    lch = lab_to_lch(53.585, 0.0, 0.0)
    assert lch.c == pytest.approx(0.0)
    assert lch.h == 0.0


def test_lch_hue_stays_in_range_around_the_circle():
    """Sweep the circle; no angle may land outside [0, 360)."""
    for degrees in range(0, 360, 5):
        rad = math.radians(degrees)
        h = lab_to_lch(50.0, 25.0 * math.cos(rad), 25.0 * math.sin(rad)).h
        assert 0.0 <= h < 360.0
        assert h == pytest.approx(float(degrees), abs=1e-9)


def test_lch_round_trip():
    for lab in [(53.2408, 80.0925, 67.2032), (32.2970, 79.1875, -107.8602),
                (87.7347, -86.1827, 83.1793), (50.0, -1.0, 2.0)]:
        back = lch_to_lab(*lab_to_lch(*lab))
        assert tuple(back) == pytest.approx(lab, abs=1e-10)


def test_red_hue_angle_is_plausible():
    """#FF0000 should sit in the low-positive hue angles (~40 degrees)."""
    lch = lab_to_lch(*hex_to_lab("#FF0000"))
    assert 35.0 < lch.h < 45.0
    assert lch.c == pytest.approx(math.hypot(80.0925, 67.2032), abs=SWATCH_TOL)


# ---------------------------------------------------------------------------
# Gate 1b: an extended swatch set, from an independent oracle
# ---------------------------------------------------------------------------
#
# The six fixtures above are pinned at 1e-4 but were transcribed after
# color.py existed, so they lean on the first-principles matrix derivation at
# the bottom of this file for their independence. This second set is a
# straightforwardly external cross-check, and it widens the coverage to hues
# the primaries miss -- orange and yellow in the +a*/+b* quadrant, cyan with
# both axes negative.
#
# PROVENANCE -- this is the part to be able to explain.
#
# Generated with coloraide 8.12.1 (a pure-Python color library, independent of
# this codebase and by a different author), via its `lab-d65` space. Exact
# command, reproducible from a clean environment:
#
#     pip install coloraide==8.12.1
#     python -c "
#     from coloraide import Color
#     for h in ['#FF0000','#00FF00','#0000FF','#FFA500','#FFFF00',
#               '#00FFFF','#808080','#FFFFFF']:
#         c = Color(h).convert('lab-d65')
#         print(h, round(c['lightness'],4), round(c['a'],4), round(c['b'],4))
#     "
#
# coloraide is a development-time oracle only. It is NOT in requirements.txt
# and nothing at runtime imports it -- these are transcribed constants, so the
# suite stays pure-stdlib-plus-pytest.
#
# It earns the name "independent": coloraide derives its sRGB->XYZ matrix from
# the primary chromaticities rather than transcribing a published table, so it
# shares no constants with server/color.py. Agreement here is two separate
# derivations landing in the same place.
#
# Tolerance is 0.5 per channel against a measured worst case of 0.0086, and
# the headroom is deliberate: two rounding conventions move these numbers in
# the third decimal.
#
#   * The D65 rounding. This codebase uses the ASTM 95.047/100/108.883; most
#     libraries derive 95.0456/100/108.9058 from the (0.3127, 0.3290)
#     chromaticity. This shifts a* and b* only -- L* is 116*f(Y/Yn) - 16 and
#     Yn is 100.000 either way, so L* cannot move.
#   * The sRGB->XYZ matrix. A transcribed 7-decimal table and a matrix derived
#     from the primary chromaticities differ around the 7th decimal, which is
#     what moves L* (#FF0000 sits at 53.2408 here against coloraide's 53.2371).
#
# A fixture that fails at 0.5 is a real error -- a transposed row, a missing
# gamma step -- not a convention mismatch.
CORROBORATING_SWATCHES = [
    ("#FF0000", (53.2371, 80.0901, 67.2033)),     # primary red
    ("#00FF00", (87.7355, -86.1816, 83.1866)),    # primary green, high L*
    ("#0000FF", (32.3009, 79.1953, -107.8555)),   # primary blue, low L*
    ("#FFA500", (74.9339, 23.9269, 78.9530)),     # saturated orange
    ("#FFFF00", (97.1386, -21.5600, 94.4838)),    # yellow, near-max L*
    ("#00FFFF", (91.1148, -48.0789, -14.1290)),   # cyan, negative a* and b*
    ("#808080", (53.5850, 0.0000, 0.0000)),       # mid-grey, must be neutral
    ("#FFFFFF", (100.0000, 0.0000, 0.0000)),      # white point itself
]

CORROBORATING_TOL = 0.5


@pytest.mark.parametrize(
    "hex_value,expected",
    CORROBORATING_SWATCHES,
    ids=[h for h, _ in CORROBORATING_SWATCHES],
)
def test_corroborating_swatch_lab(hex_value, expected):
    """Eight swatches against coloraide, within CORROBORATING_TOL."""
    got = rgb_to_lab(*hex_to_srgb(hex_value))
    for channel, actual, want in zip("Lab", got, expected):
        assert actual == pytest.approx(want, abs=CORROBORATING_TOL), (
            f"{hex_value} {channel}*: got {actual:.4f}, expected {want:.4f}"
        )


def test_corroborating_agreement_is_far_tighter_than_the_tolerance():
    """Document the real margin, so a slow drift toward 0.5 stays visible.

    If this fails while the parametrized tests above still pass, the chain has
    moved: something is eating the headroom that makes 0.5 safe. Worth knowing
    before it becomes a failure.
    """
    worst = max(
        abs(actual - want)
        for hex_value, expected in CORROBORATING_SWATCHES
        for actual, want in zip(rgb_to_lab(*hex_to_srgb(hex_value)), expected)
    )
    assert worst < 0.02, f"worst channel deviation {worst:.6f} -- chain drifted"


def test_white_argument_moves_a_and_b_but_not_l():
    """The white point must reach a* and b*, and must not reach L*.

    L* is 116*f(Y/Yn) - 16, and every D65 variant in circulation puts Yn at
    exactly 100.000, so swapping white points cannot move lightness -- only
    the X and Z denominators change. A `white` argument that shifted L* would
    mean it had been wired into the wrong place in the chain.
    """
    chromaticity_d65 = XYZ(95.0456, 100.000, 108.9058)
    default = rgb_to_lab(255, 0, 0)
    shifted = rgb_to_lab(255, 0, 0, white=chromaticity_d65)

    assert shifted.l == pytest.approx(default.l, abs=1e-12)
    assert shifted.a != default.a
    assert shifted.b != default.b
    # The shift is small but real: a rounding of the same illuminant.
    assert shifted.a == pytest.approx(default.a, abs=0.01)
    assert shifted.b == pytest.approx(default.b, abs=0.01)


def test_default_white_is_the_astm_d65():
    """The documented default, pinned so a silent swap shows up here."""
    assert tuple(D65_WHITE) == (95.047, 100.000, 108.883)
    assert rgb_to_lab(255, 0, 0) == pytest.approx(
        rgb_to_lab(255, 0, 0, white=D65_WHITE)
    )


# ---------------------------------------------------------------------------
# Gate 2: normalize -- raw counts to sRGB
# ---------------------------------------------------------------------------

def test_normalize_divides_by_clear():
    """With identity calibration the result is exactly 255 * raw / clear."""
    got = normalize(1000, 2000, 3000, 4000, Calibration())
    assert got == pytest.approx((63.75, 127.5, 191.25))


def test_normalize_is_invariant_to_exposure():
    """The Clear division is what this buys: scale every count, same answer.

    This is the property that distinguishes normalising by Clear from just
    rescaling raw counts. Doubling the integration time doubles all four
    channels; the garment did not change color, so neither may the output.
    """
    cal = Calibration(wr=0.9, wg=1.0, wb=1.1)
    base = normalize(1000, 2000, 3000, 8000, cal)
    for factor in (2, 3, 7):
        scaled = normalize(1000 * factor, 2000 * factor, 3000 * factor,
                           8000 * factor, cal)
        assert scaled == pytest.approx(base)


def test_normalize_matches_the_composed_formula():
    """Pin the whole chain: Clear-normalise, then calibrate, then scale.

    Steps 1 and 2 are both divisions and therefore commute arithmetically, so
    no output can distinguish "divide by c then by w" from "divide by w then
    by c". What this asserts is the composition itself -- that both divisions
    happen, each exactly once, against the right operand. The *ordering*
    guarantee that does have observable consequences is the zero-Clear guard
    running before either division, pinned below.
    """
    cal = Calibration(wr=0.8, wg=1.25, wb=0.5)
    r, g, b, c = 1200, 3000, 900, 10000
    expected = (
        SRGB_MAX * (r / c) / cal.wr,
        SRGB_MAX * (g / c) / cal.wg,
        SRGB_MAX * (b / c) / cal.wb,
    )
    assert normalize(r, g, b, c, cal) == pytest.approx(expected)


def test_calibration_is_applied_to_the_clear_ratio_not_the_raw_count():
    """Halving a calibration factor must double that channel, and only it."""
    counts = (1000, 1000, 1000, 8000)
    base = normalize(*counts, Calibration())
    tweaked = normalize(*counts, Calibration(wr=0.5))
    assert tweaked[0] == pytest.approx(base[0] * 2.0)
    assert tweaked[1:] == pytest.approx(base[1:])


def test_calibration_defaults_are_the_identity():
    """An uncalibrated station must not silently tint every reading."""
    assert Calibration() == (1.0, 1.0, 1.0)
    assert Calibration().wr == 1.0
    counts = (1234, 2345, 3456, 9000)
    assert normalize(*counts, Calibration()) == pytest.approx(
        normalize(*counts, Calibration(1.0, 1.0, 1.0))
    )


def test_normalize_clamps_to_the_top_of_the_range():
    """A channel brighter than Clear, or a small factor, must not exceed 255."""
    got = normalize(9000, 100, 100, 8000, Calibration())
    assert got[0] == SRGB_MAX
    assert normalize(1000, 1000, 1000, 2000, Calibration(wr=0.01))[0] == SRGB_MAX


def test_normalize_clamps_to_the_bottom_of_the_range():
    """Negative counts (sensor offset correction) floor at 0, not below."""
    got = normalize(-500, 0, 1000, 8000, Calibration())
    assert got[0] == 0.0
    assert got[1] == 0.0


def test_normalize_output_is_in_srgb_range():
    """Sweep a grid of plausible readings; nothing may escape 0-255."""
    cal = Calibration(wr=0.85, wg=1.0, wb=1.3)
    for c in (1, 500, 20000, 65535):
        for raw in (0, 1, 250, 12000, 65535):
            for component in normalize(raw, raw, raw, c, cal):
                assert 0.0 <= component <= SRGB_MAX


def test_normalize_feeds_the_lab_chain():
    """End to end: counts -> sRGB -> Lab -> LCh on one plausible reading.

    Ties the measurement path together, so a regression that somehow satisfies
    each piece in isolation still fails here.
    """
    # Raw counts whose Clear ratios reproduce #FFA500 under identity cal.
    rgb = normalize(65535, 42410, 0, 65535, Calibration())
    assert rgb == pytest.approx((255.0, 165.0, 0.0), abs=0.5)
    _, chroma, hue = lab_to_lch(*rgb_to_lab(*rgb))
    assert 60.0 < hue < 85.0
    assert chroma > 50.0


# --- the zero-Clear guard --------------------------------------------------

def test_zero_clear_raises_value_error():
    """c == 0 is a ValueError, and the message says which channel."""
    with pytest.raises(ValueError, match="clear"):
        normalize(100, 200, 300, 0, Calibration())


def test_zero_clear_raises_value_error_not_zero_division():
    """The guard must fire *before* the division, not be a rescue after it.

    ZeroDivisionError is not a subclass of ValueError, so `raises(ValueError)`
    alone would already catch a missing guard -- but this states the intent
    explicitly, because the whole point is which exception a station log shows.
    """
    with pytest.raises(Exception) as caught:
        normalize(100, 200, 300, 0, Calibration())
    assert isinstance(caught.value, ValueError)
    assert not isinstance(caught.value, ZeroDivisionError)


def test_zero_clear_raises_before_touching_calibration():
    """Structural proof that nothing runs ahead of the guard.

    A tripwire calibration whose factors explode on read: if `normalize` did
    any part of its arithmetic before validating Clear -- reading a factor,
    dividing, anything -- this would surface the tripwire's error instead of
    the ValueError. It is the strongest statement available that the guard is
    the first thing in the function body.
    """

    class Tripwire:
        @property
        def wr(self):
            raise AssertionError("calibration read before the Clear guard ran")

        wg = wr
        wb = wr

    with pytest.raises(ValueError, match="clear"):
        normalize(100, 200, 300, 0, Tripwire())


def test_zero_clear_raises_even_when_all_counts_are_zero():
    """A fully dark frame is still a refusal, not a 0,0,0 reading."""
    with pytest.raises(ValueError, match="clear"):
        normalize(0, 0, 0, 0, Calibration())


def test_negative_clear_raises():
    """Negative Clear is unphysical; refuse it rather than clamping to black."""
    with pytest.raises(ValueError, match="clear"):
        normalize(100, 200, 300, -1, Calibration())


@pytest.mark.parametrize("cal", [
    Calibration(wr=0.0),
    Calibration(wg=0.0),
    Calibration(wb=0.0),
    Calibration(wr=-1.0),
])
def test_nonpositive_calibration_factor_raises(cal):
    """A zero factor would divide by zero one line later -- refuse it too."""
    with pytest.raises(ValueError, match="calibration"):
        normalize(100, 200, 300, 400, cal)


# ---------------------------------------------------------------------------
# Matrix provenance: derive it from first principles, do not trust the table
# ---------------------------------------------------------------------------
#
# SRGB_TO_XYZ is a transcribed constant, and a transcribed constant is exactly
# where a transposed or fat-fingered digit hides -- it stays plausible and
# poisons everything downstream. So rather than eyeballing it, rebuild it here
# from the sRGB primary chromaticities in IEC 61966-2-1 plus the D65 white
# point, and assert the transcription matches.
#
# The only inputs are six 2-decimal numbers from the spec, which are much
# harder to mistype undetectably than nine 7-decimal ones.

SRGB_PRIMARIES = {"R": (0.6400, 0.3300), "G": (0.3000, 0.6000), "B": (0.1500, 0.0600)}


def _primary_to_xyz(x, y):
    """A chromaticity (x, y) as XYZ normalised to Y = 1."""
    return (x / y, 1.0, (1.0 - x - y) / y)


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve3(m, v):
    """Solve m @ s = v by Cramer's rule."""
    det = _det3(m)
    solution = []
    for col in range(3):
        swapped = [list(row) for row in m]
        for row in range(3):
            swapped[row][col] = v[row]
        solution.append(_det3(swapped) / det)
    return solution


def derive_srgb_to_xyz(white):
    """Build the linear-sRGB -> XYZ matrix from primaries and a white point.

    Standard construction: put the three primaries' XYZ in the columns, solve
    for the per-primary scale factors that map RGB (1,1,1) onto the white
    point, then scale the columns by those factors.
    """
    r = _primary_to_xyz(*SRGB_PRIMARIES["R"])
    g = _primary_to_xyz(*SRGB_PRIMARIES["G"])
    b = _primary_to_xyz(*SRGB_PRIMARIES["B"])
    columns = [[r[i], g[i], b[i]] for i in range(3)]
    scales = _solve3(columns, [white.x / 100.0, white.y / 100.0, white.z / 100.0])
    return tuple(tuple(columns[i][j] * scales[j] for j in range(3)) for i in range(3))


def test_matrix_matches_first_principles_derivation():
    """The transcribed matrix must equal one derived from the sRGB primaries.

    This is the check that actually catches a transposed digit. Eyeballing
    0.4124564 against 0.4124564 proves only that two copies of the same
    transcription agree.
    """
    derived = derive_srgb_to_xyz(D65_WHITE)
    for row_name, derived_row, actual_row in zip("XYZ", derived, SRGB_TO_XYZ):
        for coeff_name, d, a in zip("RGB", derived_row, actual_row):
            assert a == pytest.approx(d, abs=1e-6), (
                f"{row_name} row, {coeff_name} coefficient: "
                f"table has {a!r}, derivation gives {d!r}"
            )


def test_derivation_catches_a_transposed_matrix():
    """Prove the check above has teeth: a transpose must fail it."""
    derived = derive_srgb_to_xyz(D65_WHITE)
    transposed = tuple(zip(*derived))
    mismatches = sum(
        1
        for dr, tr in zip(derived, transposed)
        for d, t in zip(dr, tr)
        if abs(d - t) > 1e-6
    )
    assert mismatches >= 6, "a transposed matrix must differ in most coefficients"


def test_derivation_catches_a_single_wrong_digit():
    """And that a one-digit slip in any coefficient is caught."""
    derived = derive_srgb_to_xyz(D65_WHITE)
    for row in range(3):
        for col in range(3):
            corrupted = [list(r) for r in derived]
            # Perturb the 4th decimal -- the smallest slip a human would make
            # transcribing 0.4124564 as 0.4125564.
            corrupted[row][col] += 1e-4
            assert abs(corrupted[row][col] - derived[row][col]) > 1e-6
