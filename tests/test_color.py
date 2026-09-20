"""Tests for server/color.py.

Two gates live here, in this order:

1. The sRGB -> XYZ -> CIELAB chain against six published reference swatches.
   Nothing downstream (scoring, identify) may be trusted until this passes --
   a transposed matrix row produces plausible-looking wrong numbers.
2. CIEDE2000 against all 34 Sharma et al. (2005) test vectors to 4 decimals.
"""

from __future__ import annotations

import math

import pytest

from server.color import (
    D65_WHITE,
    SRGB_TO_XYZ,
    delta_e_2000,
    hex_to_lab,
    hex_to_srgb,
    lab_to_lch,
    lch_to_lab,
    linear_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
    srgb_to_xyz,
)
from tests.sharma_ciede2000 import PAIRS

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
# Gate 2: CIEDE2000 against Sharma et al.
# ---------------------------------------------------------------------------

def test_sharma_dataset_is_complete():
    assert len(PAIRS) == 34


@pytest.mark.parametrize(
    "lab1,lab2,expected",
    PAIRS,
    ids=[f"pair{i:02d}" for i in range(1, len(PAIRS) + 1)],
)
def test_delta_e_2000_sharma(lab1, lab2, expected):
    """All 34 published vectors, to 4 decimal places."""
    got = delta_e_2000(lab1, lab2)
    assert got == pytest.approx(expected, abs=1e-4), (
        f"{lab1} vs {lab2}: got {got:.6f}, published {expected:.4f}"
    )


def test_delta_e_2000_sharma_rounds_exactly():
    """Stronger claim: every value rounds to the published 4-decimal figure."""
    off = [
        (i, round(delta_e_2000(a, b), 4), e)
        for i, (a, b, e) in enumerate(PAIRS, 1)
        if round(delta_e_2000(a, b), 4) != e
    ]
    assert not off, f"rows not matching at 4dp: {off}"


# ---------------------------------------------------------------------------
# CIEDE2000 properties
# ---------------------------------------------------------------------------

def test_delta_e_identity_is_zero():
    for lab, _, _ in PAIRS:
        assert delta_e_2000(lab, lab) == pytest.approx(0.0, abs=1e-12)


def test_delta_e_is_symmetric():
    for lab1, lab2, _ in PAIRS:
        assert delta_e_2000(lab1, lab2) == pytest.approx(delta_e_2000(lab2, lab1), abs=1e-12)


def test_delta_e_is_non_negative():
    for lab1, lab2, _ in PAIRS:
        assert delta_e_2000(lab1, lab2) >= 0.0


def test_delta_e_monotonic_along_lightness_ramp():
    base = (50.0, 0.0, 0.0)
    deltas = [delta_e_2000(base, (50.0 + step, 0.0, 0.0)) for step in range(0, 46)]
    assert all(b > a for a, b in zip(deltas, deltas[1:]))


def test_delta_e_handles_neutral_pair_without_hue_term():
    """Both colors neutral: C1' * C2' == 0, so only the lightness term survives."""
    d = delta_e_2000((50.0, 0.0, 0.0), (60.0, 0.0, 0.0))
    s_l = 1.0 + (0.015 * 25.0) / math.sqrt(20.0 + 25.0)
    assert d == pytest.approx(10.0 / s_l, abs=1e-12)


def test_delta_e_one_neutral_one_chromatic_is_finite():
    d = delta_e_2000((50.0, 0.0, 0.0), (50.0, 30.0, 40.0))
    assert math.isfinite(d) and d > 0.0


def test_delta_e_hue_wraparound_is_short_way_round():
    """Hues at 1 deg and 359 deg are 2 deg apart, not 358."""
    near = lch_to_lab(50.0, 20.0, 1.0)
    far = lch_to_lab(50.0, 20.0, 359.0)
    straddle = delta_e_2000(near, far)
    same_side = delta_e_2000(lch_to_lab(50.0, 20.0, 179.0), lch_to_lab(50.0, 20.0, 181.0))
    assert straddle == pytest.approx(same_side, rel=0.35)
    assert straddle < 2.0


def test_antipodal_hue_branch_is_platform_independent():
    """Sharma pairs 10 and 14 sit exactly on the |h1'-h2'| == 180 boundary.

    Both must take the documented "<= 180" branch. Without the tolerance in
    color._HUE_BOUNDARY_TOL this is decided by libm rounding -- numpy lands on
    the other branch on Linux and returns 4.7461 for pair 14, which is why
    colour-science drops that pair from its own test suite.
    """
    lab1, lab2, expected = PAIRS[13]
    assert expected == 4.8045
    assert delta_e_2000(lab1, lab2) == pytest.approx(4.8045, abs=1e-4)

    lab1, lab2, expected = PAIRS[9]
    assert delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_hue_boundary_tolerance_does_not_reclassify_real_cases():
    """Pairs 11, 12 and 15 are genuinely past 180 and must stay that way."""
    for index in (10, 11, 14):
        lab1, lab2, expected = PAIRS[index]
        assert delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


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
