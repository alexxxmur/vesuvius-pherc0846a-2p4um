"""Positive control for the whole chain on a segment with known ink: PHerc0139 w043.

1. renders a 2048×4096 window of w043 with render_tifxyz_light.py from the team mesh
   `…-on-20260102150214-2.399um.tifxyz` (same volume as the team's 2.399 µm surface volume);
2. compares it slice by slice with the team's own surface volume (geometry check);
3. runs the canonical 2 µm model in both layer orders and scores it against the ink_9um labels
   (`pherc0139-w043`, annotated on the 2.4 µm render pooled 4×, so the window maps exactly).

Our result: CT r = 0.971 at mesh nodes, r = 0.993–0.997 for every slice after reversing the order,
model AUC 0.872 (official layer order) vs 0.440 (wrong order).
Labels: `hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w043 <dir>`.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/'
SEG = 'PHerc0139/segments/20260112000000-w043_2026011217'
MESH = f'{SEG}/mesh/20260112000000-on-20260102150214-2.399um.tifxyz'
TEAM_SV = f'{SEG}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr'
VOLUME = 'PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr'
Y0, Y1, X0, X1 = 16384, 18432, 8448, 12544      # window around labelled κ and ρ (level-0 px)
HERE = Path(__file__).resolve().parent


def z(a):
    a = a.astype(np.float32)
    a -= a.mean()
    return a/(a.std()+1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', required=True)
    parser.add_argument('--labels', required=True, help='dir with pherc0139-w043_inklabels.zarr and _supervision_mask.zarr')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    import tifffile
    import zarr
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(HERE))
    from render_tifxyz_light import ChunkStore
    work = Path(args.work).expanduser()
    (work/'mesh').mkdir(parents=True, exist_ok=True)
    for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'):
        if not (work/'mesh'/f).exists():
            urllib.request.urlretrieve(B+MESH+'/'+f, work/'mesh'/f)
    ours = work/'w043-ours.zarr'
    if not (ours/'render.json').exists():
        subprocess.run([sys.executable, str(HERE/'render_tifxyz_light.py'), str(work/'mesh'), VOLUME, str(ours),
                        '--seg-level-factor', '1', '--depth', '65', '--crop', str(Y0), str(Y1), str(X0), str(X1)],
                       check=True)
    team = ChunkStore(TEAM_SV, work/'team-sv-cache')
    block, _, _ = team.block(np.array([0, Y0, X0]), np.array([team.shape[0], Y1, X1]))
    o = zarr.open(str(ours/'0'), mode='r')
    for k in (2, 32, 62):
        best = max(range(block.shape[0]), key=lambda j: float((z(o[k][::2, ::2])*z(block[j][::2, ::2])).mean()))
        r = float((z(o[k][::2, ::2])*z(block[best][::2, ::2])).mean())
        print(f'our slice {k} best matches team slice {best}, r = {r:.3f}')
    ink = np.asarray(zarr.open(f'{args.labels}/pherc0139-w043_inklabels.zarr/0', mode='r')[10]) > 0
    sup = np.asarray(zarr.open(f'{args.labels}/pherc0139-w043_supervision_mask.zarr/0', mode='r')[10]) > 0
    for flag in ('', '--reverse'):
        out = work/f'w043-ours{flag.replace("--", "-")}.tif'
        subprocess.run([sys.executable, str(HERE/'canonical2um_infer.py'), str(ours), args.checkpoint, str(out),
                        '--start', '1', '--end', '63']+([flag] if flag else []), check=True)
        p = tifffile.imread(out).astype(np.float32)/255
        p4 = p.reshape(p.shape[0]//4, 4, p.shape[1]//4, 4).mean((1, 3))
        i = ink[Y0//4:Y0//4+p4.shape[0], X0//4:X0//4+p4.shape[1]]
        s = sup[Y0//4:Y0//4+p4.shape[0], X0//4:X0//4+p4.shape[1]]
        print(f'canonical model {flag or "(default order)"}: AUC {roc_auc_score(i[s], p4[s]):.3f} on {int(s.sum())} px')


if __name__ == '__main__':
    main()
