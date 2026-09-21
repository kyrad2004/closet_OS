# Closet OS — server

Flask + SQLite + color engine for the closet station.

**Status: `color.py` only.** The measurement path is implemented and
validated: raw sensor counts through normalisation, sRGB, CIELAB and LCh.
CIEDE2000, `scoring.py`, `identify.py`, the DB schema and the Flask endpoints
are not started yet — deliberately, because the color layer is a gate for
everything above it.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

53 tests, no dependencies beyond pytest. `color.py` is pure standard library.

## What `color.py` provides

| Function | Purpose |
|---|---|
| `Calibration` | namedtuple of `wr` / `wg` / `wb` white-balance factors, defaulting to the identity |
| `normalize` | raw 16-bit TCS34725 counts → 0–255 sRGB: divide by Clear, divide by the calibration factor, scale, clamp |
| `hex_to_srgb` / `srgb_to_linear` / `linear_to_srgb` | sRGB plumbing (IEC 61966-2-1) |
| `srgb_to_xyz` | linear sRGB → XYZ, D65, scaled 0–100 |
| `xyz_to_lab` | XYZ → CIE L\*a\*b\* |
| `srgb_to_lab` / `rgb_to_lab` / `hex_to_lab` | the whole chain in one call |
| `lab_to_lch` / `lch_to_lab` | cylindrical form for harmony reasoning |

### `normalize` and the zero-Clear guard

`c == 0` means no light reached the sensor. `normalize` raises `ValueError`
before any arithmetic runs, so a station log shows a named refusal rather than
a `ZeroDivisionError` from the middle of the chain. The ordering is pinned
structurally by a tripwire calibration object whose factors raise on read — the
test fails if the guard is moved even one statement later.

Note that the Clear division and the calibration division commute
arithmetically, so no output can distinguish their order; what the tests pin is
the composition, the exposure-invariance property the Clear division buys (scale
all four counts, get the same answer), and the guard ordering, which does have
observable consequences.

Still not implemented: persisting a `Calibration` — that lands with the
`calibration` table.

## Gate 1 — the Lab conversion

`tests/test_color.py` pins all six reference swatches from the handoff.

| hex | L\* | a\* | b\* |
|---|---|---|---|
| `#FFFFFF` | 100.0000 | 0.0000 | 0.0000 |
| `#000000` | 0.0000 | 0.0000 | 0.0000 |
| `#FF0000` | 53.2408 | 80.0925 | 67.2032 |
| `#00FF00` | 87.7347 | −86.1827 | 83.1793 |
| `#0000FF` | 32.2970 | 79.1875 | −107.8602 |
| `#808080` | 53.5850 | 0.0000 | 0.0000 |

Worst deviation across all 18 components: **3.6e-05**.

### Which D65?

There are two numeric D65 white points in circulation and they disagree in the
third decimal of L\*:

* **95.047 / 100.000 / 108.883** — the ASTM-rounded values, what the handoff
  specifies, and what the published Lab figures above are computed against.
  This is what we use.
* **95.0456 / 100.000 / 108.9058** — derived from the D65 chromaticity
  (0.3127, 0.3290). `colour-science` defaults to this, which is why a
  cross-check against that library shows `#FF0000` at L\* = 53.2329 rather than
  53.2408.

Neither is wrong; they are different roundings of the same illuminant. We match
the handoff and the published reference values. If you ever compare against an
online converter and see a disagreement in the second decimal, this is why.

The `SRGB_TO_XYZ` matrix rows are normalised so they sum *exactly* to that
white point. The published 7-decimal coefficients sum to 1.0000001 on the Y
row, which would put pure white at L\* = 100.000004 and give every neutral grey
an a\* of about −1.7e-05. Perceptually that is nothing, but `scoring.py` keys
its neutral-wildcard branch off `C* < 12` and `identify.py` ranks by a
chroma-sensitive distance, so an exactly-neutral axis is worth having as a
structural guarantee. After normalisation, neutrality holds to 5.6e-14 and no
reference swatch moves at 4 decimal places.

## CIEDE2000 — deferred, with its dataset retained

`delta_e_2000` is **not implemented**. It is built and validated as its own
step, against the Sharma et al. published test vectors.

`tests/sharma_ciede2000.py` — all 34 vectors — is **retained and unused**,
deliberately. It is worth more than the code it will validate: the author's
original host (`www2.ece.rochester.edu`, `hajim.rochester.edu`) is unreachable
from this build environment (DNS failure on the first, egress proxy refusal on
the second), so the table was reconstructed from two independent published
transcriptions and cross-checked field by field:

* `colour-science` 0.4.7 — `colour/difference/tests/test_delta_e.py`
* `coloraide` 8.12.1 — `tests/test_distance.py`

All 33 rows the two sources share agree to within 1e-9 on every field. They
differ only on pair 14, and in a way that resolves cleanly: `colour-science`
**excludes** it (platform-dependent `arctan2`) but records its values verbatim
in a comment, while `coloraide` includes it but typos the sample a\* as
`0.00010`. The file uses `0.0010`, which is what the colour-science comment
records and what the 0.0009 / 0.0010 / 0.0011 / 0.0012 progression across pairs
13–16 requires.

**This is a second-hand transcription, not the primary file.** If you want the
primary artifact for the writeup, fetch `CIEDE2000.txt` from Sharma's page on a
machine with open networking and diff it against `tests/sharma_ciede2000.py`.

### Known traps, for when this is built

Recorded here so they are not rediscovered the hard way. The paper names four:
`dhp` quadrant selection, `hbarp` across the 0°/360° seam, the `C1'·C2' == 0`
degenerate cases, and the sign of `RT`. A fifth it only implies: Sharma pairs 10
and 14 have hues *exactly* 180° apart, where the `|h1' - h2'| <= 180` branch
test sits on a knife edge and the answer is decided by libm rounding —
`numpy.arctan2` lands on the other branch on Linux and returns 4.7461 instead of
the published 4.8045, which is precisely why `colour-science` drops pair 14 from
its own suite. A small tolerance (~1e-10) on that comparison makes the
degenerate pairs deterministic across platforms without reclassifying anything:
the two degenerate pairs sit at 0.0 and −2.8e-14 from the boundary, and the
nearest genuinely-past-180 case (pair 11) sits at +1.5e-3.
