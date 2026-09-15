# Closet OS — server

Flask + SQLite + color engine for the closet station.

**Status: `color.py` only.** The conversion chain and CIEDE2000 are implemented
and validated. `scoring.py`, `identify.py`, the DB schema and the Flask
endpoints are not started yet — deliberately, because the handoff makes the
color layer a gate for everything above it.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

65 tests, no dependencies beyond pytest. `color.py` is pure standard library.

## What `color.py` provides

| Function | Purpose |
|---|---|
| `hex_to_srgb` / `srgb_to_linear` / `linear_to_srgb` | sRGB plumbing (IEC 61966-2-1) |
| `srgb_to_xyz` | linear sRGB → XYZ, D65, scaled 0–100 |
| `xyz_to_lab` | XYZ → CIE L\*a\*b\* |
| `srgb_to_lab` / `hex_to_lab` | the whole chain in one call |
| `lab_to_lch` / `lch_to_lab` | cylindrical form for harmony reasoning |
| `delta_e_2000` | CIEDE2000 color difference |

Not yet implemented, and intentionally so: normalising raw TCS34725 counts by
the Clear channel and applying stored white-balance factors. Those are steps 1–3
of the handoff's chain; they depend on the `calibration` table and land with it.

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

## Gate 2 — CIEDE2000

Hand-rolled from Sharma, Wu & Dalal (2005), *The CIEDE2000 color-difference
formula: Implementation notes, supplementary test data, and mathematical
observations*, Color Research & Application 30(1), 21–30. **No library is used
at runtime** — `colour-science` appears only as a development-time cross-check
and is not in `requirements.txt`.

Validated against **all 34 published test vectors**, in
`tests/sharma_ciede2000.py`. Every one agrees to 4 decimal places; worst
deviation **4.95e-05**, and all 34 round exactly to the published figure.

### Dataset provenance — read this

The author's original host (`www2.ece.rochester.edu`, `hajim.rochester.edu`)
was **unreachable from the build environment** — DNS failure on the first, and
the egress proxy refused the second. The table was therefore reconstructed from
two independent published transcriptions and cross-checked field by field:

* `colour-science` 0.4.7 — `colour/difference/tests/test_delta_e.py`
* `coloraide` 8.12.1 — `tests/test_distance.py`

All 33 rows the two sources share agree to within 1e-9 on every field. They
differ only on pair 14, and in a way that resolves cleanly:

* `colour-science` **excludes** pair 14 (platform-dependent `arctan2`), but
  records its values verbatim in a comment.
* `coloraide` includes it but typos the sample a\* as `0.00010`.

We use `0.0010`, which is what the colour-science comment records and what the
0.0009 / 0.0010 / 0.0011 / 0.0012 progression across pairs 13–16 requires.

**This is a second-hand transcription, not the primary file.** It is
corroborated by two independent sources plus an independent implementation
(below), which is strong, but if you want the primary artifact for the writeup,
fetch `CIEDE2000.txt` from Sharma's page on a machine with open networking and
diff it against `tests/sharma_ciede2000.py`.

### Independent implementation cross-check

Our `delta_e_2000` agrees with `colour-science` 0.4.7:

* **exactly (0.0 difference)** on all 34 Sharma pairs
* to **1.7e-13** over 20,000 random Lab pairs

### The antipodal-hue trap

Sharma pairs 10 and 14 have hues *exactly* 180° apart, where the formula's
`|h1' − h2'| <= 180` branch test sits on a knife edge and the answer is decided
by libm rounding. `numpy.arctan2` lands on the other branch on Linux and
returns 4.7461 instead of the published 4.8045 — which is precisely why
`colour-science` drops pair 14 from its own suite.

`color.py` applies a 1e-10 tolerance to that comparison (`_HUE_BOUNDARY_TOL`).
Measured margins in the dataset: the two degenerate pairs sit at 0.0 and
−2.8e-14 from the boundary, and the nearest genuinely-past-180 case (pair 11)
sits at +1.5e-3. The tolerance therefore has ~4 orders of magnitude of headroom
above the float noise and ~7 below any real case, so it makes the degenerate
pairs deterministic across platforms without reclassifying anything. Both
directions are pinned by tests.

All four traps the paper names — `dhp` quadrant selection, `hbarp` across the
0°/360° seam, the `C1'·C2' == 0` degenerate cases, and the sign of `RT` — are
handled explicitly and commented in place.
