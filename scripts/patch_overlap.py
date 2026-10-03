"""Fraction of each patch's tifxyz nodes lying within 10 segment voxels of another patch (overlap check):
    python patch_overlap.py PATCH_SEGMENT_DIR [PATCH_SEGMENT_DIR ...]"""
import json
import sys

import numpy as np


def main():
    import tifffile
    from scipy.spatial import cKDTree
    pts = {}
    for d in sys.argv[1:]:
        x, y, z = (tifffile.imread(f'{d}/{a}.tif') for a in 'xyz')
        v = (x > 0) & (y > 0) & (z > 0)
        pts[d] = np.stack([z[v], y[v], x[v]], 1)
    res = {}
    for a in pts:
        for b in pts:
            if a != b:
                dist, _ = cKDTree(pts[b]).query(pts[a], k=1)
                res[f'{a} within {b}'] = float((dist < 10).mean())
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
