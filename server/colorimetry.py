"""Sensor-side colorimetry for Closet OS: raw counts -> sRGB -> CIELAB -> LCh.

This is the entry point for the *measurement* path. A TCS34725 hands back four
raw 16-bit counts (red, green, blue, clear); :func:`normalize` turns those into
0-255 sRGB using the stored white-balance factors, and :func:`rgb_to_lab` /
:func:`lab_to_lch` carry that into CIELAB and its cylindrical form.

Relationship to ``server/color.py``
-----------------------------------
The sRGB -> XYZ -> Lab chain and the Lab -> LCh conversion already live in
``server/color.py``, validated against published swatches and (for dE2000)
the Sharma test vectors. This module does **not** re-derive that math -- a
second copy of the matrix is a second thing to keep correct, and the failure
mode (a transposed row) produces plausible-looking wrong numbers rather than
an exception. :func:`rgb_to_lab` and :func:`lab_to_lch` here are thin
compositions over that module, so there is exactly one implementation of the
transform in the codebase.

What is genuinely new here is :class:`Calibration` and :func:`normalize` --
steps 1-3 of the chain, which ``color.py`` deliberately left out.

Conventions
-----------
* Raw counts are the sensor's 16-bit integrate-and-hold values.
* sRGB components are 0-255 floats, IEC 61966-2-1 transfer curve.
* Lab is CIE 1976 L*a*b*, referenced to D65 / 2-degree observer.
* Hue angles are degrees in [0, 360).
"""

from __future__ import annotations

from typing import NamedTuple

from server.color import (
    D65_WHITE,
    Lab,
    LCh,
    XYZ,
    lab_to_lch,
    srgb_to_xyz,
    xyz_to_lab,
)

__all__ = [
    "Calibration",
    "Lab",
    "LCh",
    "XYZ",
    "D65",
    "SRGB_MAX",
    "normalize",
    "rgb_to_lab",
    "lab_to_lch",
]

#: D65 reference white, 2-degree observer, Y scaled to 100. Re-exported from
#: ``server.color`` so callers get the same numbers the rest of the chain uses.
D65 = D65_WHITE

#: Full-scale sRGB component.
SRGB_MAX = 255.0


class Calibration(NamedTuple):
    """Per-channel white-balance factors from a white-reference reading.

    Each factor is the Clear-normalised ratio the sensor reports for a known
    neutral card, so dividing a measurement's ratio by it maps that card back
    onto equal R=G=B. Factors default to 1.0, which is the identity transform
    -- useful for tests and for a station that has not been calibrated yet.
    """

    wr: float = 1.0
    wg: float = 1.0
    wb: float = 1.0


def normalize(r: float, g: float, b: float, c: float, cal: Calibration):
    """Convert raw sensor counts to 0-255 sRGB.

    The chain, in order:

    1. Divide each of ``r``, ``g``, ``b`` by the Clear channel ``c``. This is
       what makes the reading independent of exposure and ambient level -- two
       measurements of the same garment under different light produce the same
       ratios even though every raw count changed.
    2. Divide each ratio by its calibration factor (``cal.wr`` / ``cal.wg`` /
       ``cal.wb``), which is what pulls the sensor's spectral response and the
       LED's tint back onto neutral.
    3. Scale to 0-255 and clamp to that range.

    Steps 1 and 2 are both divisions, so arithmetically they commute; the order
    is stated because step 3's clamp does *not* commute with them, and because
    a calibration factor only means anything as a ratio against Clear.

    ``c == 0`` means the sensor saw no light at all -- a dead LED, a covered
    aperture, or an integration cycle that never ran. There is no sensible
    reading to salvage, so this raises before any arithmetic happens rather
    than letting a ZeroDivisionError surface from the middle of the chain.

    Returns floats, not ints. Rounding is the caller's decision: the Lab
    conversion downstream is happier with the full precision, and quantising
    to integers here would throw away resolution the sensor actually has.

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
        _clamp(SRGB_MAX * (raw / c) / factor)
        for raw, factor in ((r, cal.wr), (g, cal.wg), (b, cal.wb))
    )


def _clamp(value: float) -> float:
    """Clamp to the representable sRGB range."""
    if value < 0.0:
        return 0.0
    if value > SRGB_MAX:
        return SRGB_MAX
    return value


def rgb_to_lab(r: float, g: float, b: float, white: XYZ = D65) -> Lab:
    """Convert 0-255 sRGB to CIE L*a*b*.

    Three stages, all of them in ``server.color``:

    1. sRGB gamma linearisation (IEC 61966-2-1) -- ``srgb_to_linear``.
    2. The linear-sRGB -> XYZ matrix, D65 referenced -- ``srgb_to_xyz``.
    3. The XYZ -> Lab nonlinear compression -- ``xyz_to_lab``.

    ``white`` defaults to D65 (95.047, 100.000, 108.883). Note that the
    matrix in stage 2 is itself D65-referenced, so passing a different white
    here adapts only the final compression -- it is the right knob for
    comparing against a converter that quotes a different D65 rounding, not
    for a genuine change of illuminant.
    """
    return xyz_to_lab(*srgb_to_xyz(r, g, b), white=white)
