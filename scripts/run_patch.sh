#!/usr/bin/env bash
# One patch end to end: grow (Docker) -> lightweight 2.4 µm render (±64 slices) -> canonical model,
# three 62-slice depth windows x both layer orders -> overview sheet. Linux / WSL only. Usage:
#   WORK=~/fl-work CKPT=/path/r152_3ddec_v2_l5_epoch13.ckpt VILLA_INK_OPT=/path/villa/ink-detection/optimized_inference \
#     OUT=/big/disk/out bash run_patch.sh 12
set -euo pipefail
: "${WORK:?}" "${CKPT:?}" "${VILLA_INK_OPT:?}" "${OUT:?}"
WORK=$(realpath -m "$WORK"); OUT=$(realpath -m "$OUT"); CKPT=$(realpath "$CKPT")
cd "$(dirname "$0")"
SEED=$1
SCAN=PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr
mkdir -p "$OUT"
[ -f "$WORK/seeds.json" ] || python grow_patch.py seeds --work "$WORK"
SEG=$(ls -d "$WORK"/patches/s$SEED-*-g100/auto_grown_* 2>/dev/null | tail -1 || true)
if [ -z "$SEG" ]; then
  python grow_patch.py grow --work "$WORK" --seed-index "$SEED" --gens 100
  SEG=$(ls -d "$WORK"/patches/s$SEED-*-g100/auto_grown_* | tail -1)
fi
SV="$OUT/s$SEED-d129.zarr"
for attempt in 1 2 3; do       # chunks fetched by a failed attempt stay in the cache
  [ -f "$SV/render.json" ] && break
  python render_tifxyz_light.py "$SEG" "$SCAN" "$SV" --depth 129 --workers 8 --cache "$OUT/chunk-cache" || echo "render attempt $attempt failed"
done
[ -f "$SV/render.json" ] || { echo "render failed 3 times" >&2; exit 1; }
for start in 1 33 66; do
  for flag in "" "--reverse"; do
    python canonical2um_infer.py "$SV" "$CKPT" "$OUT/s$SEED-w$start${flag:+-reverse}.tif" \
      --start $start --end $((start+62)) --stride 128 $flag
  done
done
python make_sheets.py "$OUT" "s$SEED"
