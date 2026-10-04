"""Label-free check that a surface volume actually sits on a papyrus sheet.

For a surface volume (axis 0 = offset along the surface normal) sample N random 128×128 chunk columns,
average CT over each column per depth slice and build the depth profile p(d). A mesh that follows a sheet
shows a peak (dense papyrus) above the flanks (gaps / neighbouring windings); a mesh that cuts across
windings gives a flat profile. Reports

    S        = (max p - median p) / median p of the patch-average profile (after ~10 µm smoothing)
    S_local  = median over columns of the same quantity per 128×128 column
    peak_offset_sd_um = spread of the per-column peak depth (how much the mesh wanders off the sheet)

Use S_local and the offset spread: the patch-average S also drops when a good mesh merely drifts slowly
in depth (on PHerc0139 w043 a 100 µm tilt across 1 cm drops S from 0.42 to 0.08 while the ink model's AUC
stays at 0.87), whereas S_local does not.

Works on local zarr surface volumes and on raw (uncompressed) surface volumes in the open-data bucket.

    python qc_sheet_profile.py PATH_OR_BUCKET_KEY [...] --voxel-um 2.403 --out qc.jsonl
    python qc_sheet_profile.py --from-metadata labelled --resolution 2.4 --out qc.jsonl
    python qc_sheet_profile.py out/s12-d129.zarr --voxel-um 2.403 --half-depth-um 131 --out qc.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent


def open_volume(source, cache):
    """Return (shape, read_column(y, x) -> (D,128,128) array)."""
    if Path(source).exists():
        import zarr
        a = zarr.open(str(Path(source)/'0'), mode='r')
        return a.shape, lambda y, x: np.asarray(a[:, y:y+128, x:x+128])
    sys.path.insert(0, str(HERE))
    from render_tifxyz_light import ChunkStore
    store = ChunkStore(source.rstrip('/'), Path(cache)/source.strip('/').replace('/', '_'))
    if store.chunk[0] != store.shape[0]:
        raise ValueError(f'{source}: chunks do not span the full depth {store.chunk}')
    return store.shape, lambda y, x: store.block(np.array([0, y, x]), np.array([store.shape[0], y+128, x+128]))[0]


def profile(source, voxel_um, n=40, seed=0, cache='~/.cache/vc-chunks', half_depth_um=None):
    shape, read_full = open_volume(source, str(Path(cache).expanduser()))
    depth, height, width = shape
    if half_depth_um:                      # compare surfaces over the same physical depth window
        keep = min(depth, 2*int(round(half_depth_um/voxel_um))+1)
        lo = (depth-keep)//2
        read = lambda y, x, f=read_full: f(y, x)[lo:lo+keep]
        depth = keep
    else:
        read = read_full
    rng = np.random.default_rng(seed)
    win = max(1, int(round(10/voxel_um)))
    acc, cnt, tries, local_s, local_off = np.zeros(depth), 0, 0, [], []
    while cnt < n and tries < n*12:
        tries += 1
        y = int(rng.integers(0, max(1, height//128)))*128
        x = int(rng.integers(0, max(1, width//128)))*128
        col = read(y, x).astype(np.float32)
        if col.shape[1:] != (128, 128):
            continue
        mask = col[depth//2] > 0
        if mask.mean() < .9:
            continue
        pc = col[:, mask].mean(1)
        acc += pc
        cnt += 1
        # Per-column (0.3 mm at 2.4 µm) profile: insensitive to offsets that drift slowly across the patch.
        pcs = uniform_filter1d(pc, win, mode='nearest')
        med_c = float(np.median(pcs))
        kc = int(np.argmax(pcs))
        local_s.append((pcs[kc]-med_c)/med_c)
        local_off.append((kc-(depth-1)/2)*voxel_um)
    if cnt == 0:
        return None
    p = acc/cnt
    ps = uniform_filter1d(p, win, mode='nearest')
    med = float(np.median(ps))
    k = int(np.argmax(ps))
    return {'source': str(source), 'voxel_um': voxel_um, 'depth': depth, 'columns': cnt,
            'S': float((ps[k]-med)/med), 'peak_offset_um': float((k-(depth-1)/2)*voxel_um),
            'S_local_median': float(np.median(local_s)), 'S_local_q25': float(np.percentile(local_s, 25)),
            'peak_offset_sd_um': float(np.std(local_off)), 'window_half_um': float((depth-1)/2*voxel_um),
            'profile': [round(float(v), 2) for v in p]}


def voxel_from_name(path):
    m = re.search(r'(\d+\.\d+)um', path)
    return float(m.group(1)) if m else None


def from_metadata(kind, resolution, tolerance):
    import gzip
    import urllib.request
    raw = urllib.request.urlopen('https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json', timeout=120).read()
    meta = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
    out = []
    for sid, sample in meta['samples'].items():
        for seg in sample['segments'].values():
            types = {d['type'] for d in seg.get('data', [])}
            labelled = 'ink-labels' in types
            auto = 'auto_grown' in seg['long_id']
            if (kind == 'labelled' and not labelled) or (kind == 'auto' and (labelled or not auto)):
                continue
            for d in seg.get('data', []):
                if d['type'] != 'layers-zarr':
                    continue
                for o in d['origins']:
                    v = voxel_from_name(o['path'].split('/')[-2])
                    if v and abs(v-resolution) <= tolerance:
                        out.append((sid, seg['long_id'], o['path'], v))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sources', nargs='*')
    parser.add_argument('--voxel-um', type=float, help='for local sources (bucket keys carry it in their name)')
    parser.add_argument('--from-metadata', choices=('labelled', 'auto'))
    parser.add_argument('--resolution', type=float, default=2.4)
    parser.add_argument('--tolerance', type=float, default=0.25)
    parser.add_argument('--columns', type=int, default=40)
    parser.add_argument('--half-depth-um', type=float, help='crop every volume to the central ±this many µm')
    parser.add_argument('--cache', default='~/.cache/vc-chunks')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    jobs = [(None, None, s, args.voxel_um or voxel_from_name(s)) for s in args.sources]
    if args.from_metadata:
        jobs += from_metadata(args.from_metadata, args.resolution, args.tolerance)
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)['source'] for line in open(args.out)}
    with open(args.out, 'a') as handle:
        for sid, seg, source, voxel in jobs:
            if source in done:
                continue
            try:
                r = profile(source, voxel, n=args.columns, cache=args.cache, half_depth_um=args.half_depth_um)
            except Exception as error:
                print(f'skip {source}: {error}', flush=True)
                continue
            if r is None:
                continue
            r.update({'sample': sid, 'segment': seg})
            handle.write(json.dumps(r)+'\n')
            handle.flush()
            print(f"{(seg or source)[:40]:40s} {voxel:.3f} um  S={r['S']:.3f}  S_local={r['S_local_median']:.3f}  offset_sd={r['peak_offset_sd_um']:.0f} um", flush=True)


if __name__ == '__main__':
    main()
