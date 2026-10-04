"""Overview sheet (rows: depth windows 1/33/66, columns: default/reverse layer order) and statistics
for one rendered patch:  python make_sheets.py OUT_DIR s12"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    import tifffile
    import zarr
    from PIL import Image
    out, name = Path(sys.argv[1]), sys.argv[2]
    rows, stats = [], {}
    sv = zarr.open(str(out/f'{name}-d129.zarr'/'0'), mode='r')
    # Denominator for every fraction: pixels whose rendered CT is non-zero (inside the mesh and the scan).
    ct = np.asarray(sv[sv.shape[0]//2]) > 0
    stats['ct_covered_cm2'] = float(ct.mean()*sv.shape[1]*sv.shape[2]*(2.403e-4)**2)
    for start in (1, 33, 66):
        cells = []
        for suffix in ('', '-reverse'):
            a = tifffile.imread(out/f'{name}-w{start}{suffix}.tif')
            if a.shape != ct.shape:
                raise SystemExit(f'map {a.shape} and render {ct.shape} differ')
            v = a[ct]/255
            stats[f'w{start}{suffix or "-default"}'] = {'frac_p_ge_0_5': float((v >= .5).mean()),
                                                        'p99': float(np.percentile(v, 99))}
            k = max(1, a.shape[0]//1000)
            h, w = a.shape[0]//k*k, a.shape[1]//k*k
            cells.append(np.pad(a[:h, :w].reshape(h//k, k, w//k, k).mean((1, 3)).astype(np.uint8), 6,
                                constant_values=128))
        rows.append(np.concatenate(cells, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out/f'{name}-sheet.png')
    (out/f'{name}-stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
