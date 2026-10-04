# PHerc0846A at 2.4 µm: an inconclusive First Letters search, a lightweight renderer and a sheet-profile check

**In one paragraph.** We grew three automatic patches on PHerc0846A's 2.403 µm scan (≈32 cm² with CT
data), rendered them with ±154 µm of depth and ran the canonical 2 µm ink model in three depth windows and
both layer orders. In a blind visual review of all 18 maps we saw **no letter-like shapes, rows or strokes**.
The same chain gives AUC 0.87 on the labelled letters of PHerc0139 w043, equal to the model on the team's
own render. But our patches score low on a simple sheet-profile check (one of them lower than any team
surface we measured), so this is **weak evidence**: the model shows no text on these automatic meshes; it
does not show that the band has no ink. Along the way: a lightweight tifxyz renderer, validated slice by
slice against a team surface volume, that rendered this scan where `vc_render_tifxyz` crashed.

All statements as of 4 Oct 2026.

## Why PHerc0846A

- [Bullo27/first-letters-survey](https://github.com/Bullo27/first-letters-survey) ran the released
  `ink_9um` models on all 21 eligible scrolls at ~9 µm: speckle only.
- [ShribyrLabs](https://github.com/ShribyrLabs/vesuvius-reports) measured that downsampling 2.4 µm layers
  to 9 µm drops the canonical model's letter-scale score from 0.95 to 0.11.
- In the open-data bucket only two eligible scrolls have a 2.4 µm scan: PHerc1203 (reported negative by
  [TAUIL-Abd-Elilah](https://github.com/TAUIL-Abd-Elilah/cross-scan-ink-transfer)) and **PHerc0846A**,
  for which we found no public report. Its scan is a ~36 mm band (15137 slices) at full width.

## Pipeline

| Step | Tool | Notes |
| --- | --- | --- |
| Seeds | `grow_patch.py seeds` | scan level-5 mask + team m7-L2 surface prediction (same 32-voxel grid); 18 seeds at 0.3/0.5/0.7 band height |
| Grow | `vc_grow_seg_from_seed` in `ghcr.io/scrollprize/villa/volume-cartographer@sha256:bad516f66001abca759454cc43e4fd11e5b19aa55d36bdc2043817291c8083c4` | GrowPatch parameters as in Bullo27, 100 generations requested (s12 reached 100; s0 stopped at 83, s6 at 80), level-2 (9.612 µm) coordinates |
| Render | `render_tifxyz_light.py` | level 0 (2.403 µm), `--seg-level-factor 4`, 129 slices (±64 voxels = ±154 µm) |
| Ink | `canonical2um_infer.py` | `scrollprize/ink_canonical_2um`, villa `f637f3b3` optimized_inference unchanged, 62-slice windows starting at slices 1/33/66, both orders, 256-px tiles, stride 128 |
| Review | `make_sheets.py` | blind review of prediction-only sheets first, numbers afterwards |
| Surface check | `qc_sheet_profile.py` | see below |

## Result on PHerc0846A

| Patch (seed, band height) | CT-covered area | p ≥ 0.5, default order (windows 1/33/66) | reverse order |
| --- | ---: | --- | --- |
| s12 (upper) | 14.3 cm² | 2.8 / 3.0 / 2.6 % | 3.3 / 3.3 / 4.0 % |
| s6 (middle) | 9.3 cm² | 6.9 / 5.8 / 5.8 % | 9.3 / 9.8 / 8.2 % |
| s0 (lower) | 10.1 cm² | 8.0 / 8.2 / 9.3 % | 9.3 / 9.8 / 7.9 % |

Fractions are over pixels whose rendered CT is non-zero. s0 and s6 overlap: ≈18 % of their mesh nodes lie
within 10 level-2 voxels (≈96 µm) of the other patch (`patch_overlap.py`; this proximity test could also
count an adjacent winding). Subtracting that gives ≈32 cm², a few percent of the 2.4 µm band. All maps show
rounded blobs of 0.3–0.5 mm, radial streaks that follow the mesh geometry, and fibre striping. The reverse
order is at least as noisy as the default one.

![s12 maps](results/figures/s12-sheet.jpg)

Rows: depth windows 1/33/66; columns: default/reverse order (downscaled JPGs of the `make_sheets.py`
output). Also [s6](results/figures/s6-sheet.jpg), [s0](results/figures/s0-sheet.jpg) and the
[middle CT slice of s12](results/figures/s12-ct-middle-slice.jpg). Numbers: `results/s*-stats.json`,
[results/summary.json](results/summary.json).

## Positive control: PHerc0139 w043 (`validate_w043.py`)

w043 has letters labelled in the [ink_9um dataset](https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um)
(`pherc0139-w043`, annotated on the 2.4 µm render pooled 4×, so a level-0 window maps exactly onto it).
On a 2048×4096 window around the labelled κ and ρ (133 672 supervised pixels on the 9.6 µm label grid,
61 960 of them ink), stride 128, mirrored 62-slice windows (ours 1–62 reversed = team 24–85):

| Render | AUC, right layer order | AUC, wrong order |
| --- | ---: | ---: |
| team surface volume | 0.872 | 0.447 |
| ours, from the team mesh `…-on-20260102150214-2.399um.tifxyz` | 0.872 | 0.439 |

Our render vs the team's, same window: CT sampled at the mesh nodes inside it (≤400 sampled) vs the team's
middle slice r = 0.93; each of our 65 slices k matches team slice 86 − k, with r between 0.993 and 0.994
(same geometry and spacing, **opposite layer order**). The text row and the tail of ρ appear only in the
right order:

![our render of w043](results/figures/w043-ours-render-fwd-rev.png)

The model card does not document the training data, and PHerc0139 surfaces may well be in it, so this tests
the chain (render geometry, layer order, model input/output), not transfer to an unseen scroll. A pixel AUC
on labelled strokes is also not the same as a papyrologist reading the text.

## Surface quality: `qc_sheet_profile.py`

A mesh that follows a sheet shows a CT-intensity peak along the normal (dense papyrus between gaps or
neighbouring windings). We sample 40 random 0.3 mm columns of a surface volume and report **S_local**, the
median over columns of (max − median)/median of each column's depth profile after ~10 µm smoothing. Results for the ~2.4 µm surface
volumes in the open-data bucket (2.215–2.403 µm, cropped to the central ±131 µm; our patches with
`--half-depth-um 131`):

![qc](results/figures/qc_sheet_profile.png)

| Surfaces | n | S_local median | min |
| --- | ---: | ---: | ---: |
| team, with ink labels | 27 | 0.40 | 0.19 |
| team, `auto_grown` (PHerc0814 and PHerc0009B only) | 31 | 0.22 | 0.12 |
| **ours, PHerc0846A s12 / s6 / s0** | 3 | 0.11 / 0.20 / 0.20 | |

s12 is lower than every team surface; s6 and s0 are below all ink-labelled surfaces but one and in the lower
third of the `auto_grown` ones (7 and 10 of 31 score lower). Caveats:

- S_local also tracks the scan's sheet contrast (PHercParis4 0.24–0.38 vs PHerc0139/1667 0.40–0.66), and
  there is no team surface on PHerc0846A to compare with.
- **S_local was chosen after a controlled test**, `tilt_experiment.py`. The pre-registered measure was the
  patch-average S, and its pre-registered criterion (S and the model's AUC falling together as the w043 mesh
  is tilted) failed:

  | Tilt T across the 1 cm window | 0 | 50 µm | 100 µm | 200 µm | 400 µm |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | patch-average S | 0.42 | 0.28 | 0.08 | 0.06 | 0.04 |
  | S_local | 0.53 | 0.55 | 0.63 | 0.83 | 0.31 |
  | model AUC | 0.87 | 0.84 | 0.87 | 0.84 | 0.68 |

  A slow depth drift flattens the patch-average profile while the AUC stays high. S_local is not monotonic
  either; it drops only once the sheet leaves the rendered window (T = 400 µm).
- We also computed the spread of the per-column peak depth. We do not use it: team values (median 69–77 µm)
  are close to the ≈75 µm that random peak positions would give over this window.
- The single sign chosen for s12's normals splits its nodes 50/50 against the image-centre radial direction:
  half of that patch runs roughly side-on to the radius, another sign of awkward geometry.

## The lightweight renderer

`vc_render_tifxyz` from the image above has no `--flip-normals`. On a 0.5 cm² test patch of this scan it
crashed with `corrupted size vs. prev_size` and hung after pulling ~31 GB. `render_tifxyz_light.py` finished
the same patch with 4.7 GB of downloads (65 slices, an earlier version of the script: 11 min serially; 41 s
from cache with 6 worker processes, bit-identical output). It

- upsamples the tifxyz grid to level-0 pixels (node i at pixel i/scale, the VC convention; the valid mask
  uses the same mapping);
- takes normals from the grid and orients the whole patch with **one** sign (cosine-weighted vote over valid
  nodes, away from the image centre);
- for each 256-px output tile fetches only the raw 128³ chunks inside the bounding box of that tile's
  ±N-slice shell (disk cache), renders tiles in parallel processes, and writes a surface-volume zarr in the
  published layout.

Its slice order is the reverse of `vc_render_tifxyz` on w043, so run inference in both orders. It runs on
Linux / WSL only (it uses the `fork` start method) and needs zarr ≥ 3. Level-2 segment coordinates are scaled
×4 without a half-voxel offset (≤1.5 level-0 voxels, ≈3.6 µm).

**Bugs found in review and fixed before release:**
- An earlier version chose the normal sign per mesh node. On our patches that flipped the depth order on
  50 % (s12), 18 % (s6) and 10 % (s0) of the nodes. All three patches were re-rendered and all 18 maps re-run;
  every number above comes from the fixed run. (w043 was unaffected: r ≥ 0.99 for every slice either way.)
- The valid mask previously used corner-aligned interpolation, offset by up to half a grid step (≈24 µm) at
  patch borders. The released code uses the coordinate mapping. Our renders used the old mask, which affects
  only border pixels.

## Costs

RTX 5070 Ti 16 GB, Ryzen 7 7800X3D, 30 GB RAM (WSL2), home connection ~26 MB/s.

- **Grow:** s12 22 min, s0 37 min, s6 79 min; the three grows overlapped and shared the CPU. Keep the
  tracer's work directory on a native Linux filesystem. With normal grids on a Windows drive bind-mounted into
  Docker Desktop we measured 50 generations in 3 h (<1 core busy, 65 % grid-cache hits). On ext4, 15
  generations took 22 s.
- **Render:** 25–45 min per patch at 129 slices; 27–93 GB downloaded per patch, ≈190 GB in total. s0 cut more
  chunks than its area suggests; s6 reused chunks cached for the overlapping s0; s12's figure includes one
  failed first attempt. Re-rendering from cache after the fix took ~20 min per patch.
- **Model:** 10–15 min per window and order. 18 maps ≈ 4 h, done twice (before and after the fix).
- **Total:** roughly 13 h of machine time including the rerun.

## Limitations and deviations

- Automatic meshes only, with no human check of the inked layer. One ink model. ≈32 cm².
- The patches score low on the sheet-profile check, so a negative here does not mean the band has no ink.
- Planned but not run: the shuffled-block and off-papyrus controls (there were no candidates to test them on)
  and a pooled-9.6 µm `ink_9um` comparison.
- Rendering moved from `vc_render_tifxyz` with 65 slices to our renderer with 129 slices after the crash.
- The surface-check measure changed after the tilt test (above).
- The probability maps (~40 MB each) are not in the repository. The meshes are (`results/meshes/`), so the
  maps can be regenerated, subject to the pinned tools.

## Reproduce

```bash
pip install -r requirements.txt     # plus torch with CUDA and villa optimized_inference requirements
export WORK=~/fl-work OUT=/big/disk/fl-out \
       CKPT=/path/r152_3ddec_v2_l5_epoch13.ckpt \
       VILLA_INK_OPT=/path/villa/ink-detection/optimized_inference   # villa at f637f3b3
bash scripts/run_patch.sh 12        # also 6 and 0
# to reuse our meshes instead of growing: mkdir -p $WORK/patches/s12-ours-g100 &&
#   cp -r results/meshes/s12 $WORK/patches/s12-ours-g100/auto_grown_ours   (same for s6, s0)
# w043 labels: hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w043 <dir>
python scripts/validate_w043.py --work ~/w043-check --labels <dir> --checkpoint "$CKPT"
python scripts/tilt_experiment.py --work ~/w043-check --labels <dir> --checkpoint "$CKPT"
python scripts/qc_sheet_profile.py --from-metadata labelled --resolution 2.4 --half-depth-um 131 --out labelled.jsonl
python scripts/qc_sheet_profile.py --from-metadata auto --resolution 2.4 --half-depth-um 131 --out auto.jsonl
python scripts/qc_sheet_profile.py "$OUT"/s12-d129.zarr "$OUT"/s6-d129.zarr "$OUT"/s0-d129.zarr \
       --voxel-um 2.403 --half-depth-um 131 --out ours.jsonl
```

The team-segment QC files in `results/qc/` were computed without `--half-depth-um`. Team surface volumes are
109 slices (±130 µm) or, for the one 2.215 µm volume, 118 slices (±129 µm), so the ±131 µm crop leaves them
unchanged.

## Credits and licences

- Code in this repository: MIT (`LICENSE`). Seed placement, grid prefetching and GrowPatch parameters in
  `grow_patch.py` are adapted from [Bullo27/first-letters-survey](https://github.com/Bullo27/first-letters-survey),
  MIT, notice in `LICENSE-THIRD-PARTY`.
- Tools: [ScrollPrize/villa](https://github.com/ScrollPrize/villa) (VC3D, optimized_inference).
  Model: [scrollprize/ink_canonical_2um](https://huggingface.co/scrollprize/ink_canonical_2um) (MIT per its
  model card). Labels: [scrollprize ink_9um dataset](https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um).
- CT scans, surface predictions, team meshes and surface volumes: Vesuvius Challenge open data
  ([scrollprize.org/data](https://scrollprize.org/data)), licensed
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The figures and meshes in `results/` are
  derived from it and carry the same licence.
