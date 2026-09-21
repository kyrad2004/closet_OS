"""Tests for server/colorimetry.py -- the sensor-side half of the color chain.

Two gates live here:

1. ``rgb_to_lab`` against published CIELAB values for eight hex swatches
   spanning the hue circle and the lightness range. See PROVENANCE below.
2. ``normalize`` against the raw-counts contract: Clear-normalise first,
   then calibration, then clamp -- and refuse a zero Clear reading up front
   rather than dividing by it.
"""

from __future__ import annotations

import math

import pytest

from server.colorimetry import (
    D65,
    SRGB_MAX,
    Calibration,
    lab_to_lch,
    normalize,
    rgb_to_lab,
)

# Tolerance per Lab channel for the swatch fixtures.
#
# 0.5 is loose relative to the agreement we actually get -- the worst channel
# across all eight swatches is 0.0086 -- and that headroom is deliberate. Two
# conventions in circulation move these numbers in the third decimal:
#
#   * The D65 rounding. This codebase uses the ASTM 95.047/100/108.883; most
#     libraries derive 95.0456/100/108.9058 from the (0.3127, 0.3290)
#     chromaticity. This shifts a* and b* only -- L* is 116*f(Y/Yn) - 16 and
#     Yn is 100.000 either way, so L* cannot move.
#   * The sRGB->XYZ matrix. A transcribed 7-decimal table and a matrix derived
#     from the primary chromaticities differ around the 7th decimal, which is
#     what moves L* (#FF0000 sits at 53.2408 here against coloraide's 53.2371).
#
# Both are roundings of the same colorimetry. A fixture that fails at 0.5 is a
# real error -- a transposed matrix row, a missing gamma step -- not a
# convention mismatch. test_swatch_agreement_is_far_tighter_than_the_tolerance
# below keeps the slack from quietly being spent.
LAB_TOL = 0.5

# Published CIE L*a*b* (D65, 2-degree observer) for each swatch.
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
# test suite stays pure-stdlib-plus-pytest.
#
# Two independent reasons to trust these beyond "a library said so":
#
#   1. coloraide derives its sRGB->XYZ matrix from the IEC 61966-2-1 primary
#      chromaticities rather than transcribing a published table, so it shares
#      no constants with server/color.py. Agreement here is two separate
#      derivations landing in the same place.
#   2. The four primary/secondary rows match the classic published figures
#      (the ones quoted in the README table and reproduced by essentially
#      every sRGB->Lab converter): #FF0000 at L*=53.24, a*=80.09, b*=67.20,
#      #00FF00 at 87.73/-86.18/83.18, #0000FF at 32.30/79.19/-107.86.
#
# The residual against these figures is at most 0.0086 (on #00FFFF's a*), and
# comes from the two rounding conventions described at LAB_TOL above.
REFERENCE_SWATCHES = [
    # (hex, (L*, a*, b*), what it pins)
    ("#FF0000", (53.2371, 80.0901, 67.2033)),     # primary red
    ("#00FF00", (87.7355, -86.1816, 83.1866)),    # primary green, high L*
    ("#0000FF", (32.3009, 79.1953, -107.8555)),   # primary blue, low L*
    ("#FFA500", (74.9339, 23.9269, 78.9530)),     # saturated orange
    ("#FFFF00", (97.1386, -21.5600, 94.4838)),    # yellow, near-max L*
    ("#00FFFF", (91.1148, -48.0789, -14.1290)),   # cyan, negative a* and b*
    ("#808080", (53.5850, 0.0000, 0.0000)),       # mid-gray, must be neutral
    ("#FFFFFF", (100.0000, 0.0000, 0.0000)),      # white point itself
]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``"#RRGGBB"`` into an 0-255 triple.

    Deliberately local and three lines long: these tests are the gate on
    ``rgb_to_lab``, so the fixture path into it should not route through
    another unit that could itself be wrong.
    """
    text = value.lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


# ---------------------------------------------------------------------------
# Gate 1: rgb_to_lab against published swatches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hex_value,expected",
    REFERENCE_SWATCHES,
    ids=[h for h, _ in REFERENCE_SWATCHES],
)
def test_reference_swatch_lab(hex_value, expected):
    """Each published swatch, within LAB_TOL on every channel."""
    got = rgb_to_lab(*_hex_to_rgb(hex_value))
    for channel, actual, want in zip("Lab", got, expected):
        assert actual == pytest.approx(want, abs=LAB_TOL), (
            f"{hex_value} {channel}*: got {actual:.4f}, expected {want:.4f}"
        )


def test_swatch_agreement_is_far_tighter_than_the_tolerance():
    """Document the real margin, so a slow drift toward LAB_TOL is visible.

    If this starts failing while the parametrized tests above still pass, the
    chain has moved: something is eating the headroom that makes the 0.5
    tolerance safe. That is worth knowing before it becomes a failure.
    """
    worst = max(
        abs(actual - want)
        for hex_value, expected in REFERENCE_SWATCHES
        for actual, want in zip(rgb_to_lab(*_hex_to_rgb(hex_value)), expected)
    )
    assert worst < 0.02, f"worst channel deviation {worst:.6f} -- chain drifted"


def test_black_is_the_lab_origin():
    """Zero sRGB sits at the origin, exercising the linear branch of f(t)."""
    l, a, b = rgb_to_lab(0, 0, 0)
    assert (l, a, b) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_white_lands_on_the_white_point():
    """255,255,255 must be L*=100 with no chroma, not merely close to it."""
    l, a, b = rgb_to_lab(255, 255, 255)
    assert l == pytest.approx(100.0, abs=1e-9)
    assert math.hypot(a, b) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("level", [0, 32, 64, 128, 192, 255])
def test_neutrals_have_no_chroma(level):
    """r == g == b must produce a* == b* == 0 at every lightness."""
    _, a, b = rgb_to_lab(level, level, level)
    assert math.hypot(a, b) == pytest.approx(0.0, abs=1e-9)


def test_lightness_is_monotonic_in_gray_level():
    """L* must increase with gray level -- catches a sign or branch error."""
    lightness = [rgb_to_lab(v, v, v)[0] for v in range(0, 256, 15)]
    assert all(lo < hi for lo, hi in zip(lightness, lightness[1:]))


def test_white_argument_moves_a_and_b_but_not_l():
    """The white point must reach a* and b*, and must not reach L*.

    L* is 116*f(Y/Yn) - 16, and every D65 variant in circulation puts Yn at
    exactly 100.000, so swapping white points cannot move lightness -- only
    the X and Z denominators change. A `white` argument that shifted L* would
    mean it had been wired into the wrong place in the chain.
    """
    chromaticity_d65 = type(D65)(95.0456, 100.000, 108.9058)
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
    assert tuple(D65) == (95.047, 100.000, 108.883)
    assert rgb_to_lab(255, 0, 0) == pytest.approx(rgb_to_lab(255, 0, 0, white=D65))


# ---------------------------------------------------------------------------
# Gate 2: normalize
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


# --- the zero-Clear guard --------------------------------------------------

def test_zero_clear_raises_value_error():
    """c == 0 is a ValueError, and the message says which channel."""
    with pytest.raises(ValueError, match="clear"):
        normalize(100, 200, 300, 0, Calibration())


def test_zero_clear_raises_value_error_not_zero_division():
    """The guard must fire *before* the division, not be a rescue after it.

    ZeroDivisionError is not a subclass of ValueError, so a `raises(ValueError)`
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
# lab_to_lch
# ---------------------------------------------------------------------------

def test_lch_chroma_is_the_ab_magnitude():
    """C* is hypot(a*, b*) and L* passes through untouched."""
    l, c, h = lab_to_lch(53.2371, 80.0901, 67.2033)
    assert l == pytest.approx(53.2371)
    assert c == pytest.approx(math.hypot(80.0901, 67.2033))


@pytest.mark.parametrize("a,b,expected_h", [
    (10.0, 0.0, 0.0),      # +a* axis
    (0.0, 10.0, 90.0),     # +b* axis
    (-10.0, 0.0, 180.0),   # -a* axis
    (0.0, -10.0, 270.0),   # -b* axis, must wrap up not report -90
])
def test_lch_hue_angles_on_the_axes(a, b, expected_h):
    """Hue is degrees measured from +a*, wrapped into [0, 360)."""
    assert lab_to_lch(50.0, a, b).h == pytest.approx(expected_h)


def test_lch_hue_stays_in_range_around_the_circle():
    """Sweep the circle; no angle may land outside [0, 360)."""
    for degrees in range(0, 360, 5):
        rad = math.radians(degrees)
        h = lab_to_lch(50.0, 25.0 * math.cos(rad), 25.0 * math.sin(rad)).h
        assert 0.0 <= h < 360.0
        assert h == pytest.approx(float(degrees), abs=1e-9)


def test_neutral_has_zero_chroma_and_a_defined_hue():
    """A neutral's hue is undefined; report 0.0 rather than NaN or a crash."""
    assert lab_to_lch(53.585, 0.0, 0.0) == pytest.approx((53.585, 0.0, 0.0))


def test_measured_orange_lands_in_the_orange_hue_sector():
    """End to end: counts -> sRGB -> Lab -> LCh puts orange near 73 degrees.

    Ties the three public functions together on one plausible reading, so a
    unit-level regression that somehow satisfies each piece in isolation still
    fails here.
    """
    # Raw counts whose Clear ratios reproduce #FFA500 under identity cal.
    r, g, b, c = 65535, 42410, 0, 65535
    rgb = normalize(r, g, b, c, Calibration())
    assert rgb == pytest.approx((255.0, 165.0, 0.0), abs=0.5)
    _, chroma, hue = lab_to_lch(*rgb_to_lab(*rgb))
    assert 60.0 < hue < 85.0
    assert chroma > 50.0
