"""Controlled test of qc_sheet_profile.py: tilt a good mesh (PHerc0139 w043) so it cuts across windings and
watch both the sheet salience S and the canonical model's AUC on the labelled letters.

The mesh is displaced along its own normal by a linear ramp across the 4096-px validation window, from
-T/2 to +T/2 (T = 0, 50, 100, 200, 400 µm). Run after validate_w043.py (it reuses its --work directory).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate_w043 import VOLUME, X0, X1, Y0, Y1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tilts-um', default='0,50,100,200,400')
    args = parser.parse_args()
    import tifffile
    import zarr
    from sklearn.metrics import roc_auc_score
    from render_tifxyz_light import fill_invalid, normals
    from qc_sheet_profile import profile
    work = Path(args.work).expanduser()
    x, y, z = (tifffile.imread(work/'mesh'/f'{a}.tif').astype(np.float64) for a in 'xyz')
    valid = (x > 0) & (y > 0) & (z > 0)
    pts = fill_invalid(np.stack([z, y, x], -1), valid)
    n = normals(pts, (0, 0))                           # sign irrelevant here: same ramp sign everywhere
    voxel = 2.399
    cols = np.arange(x.shape[1])*20                    # node column -> level-0 pixel
    ramp = np.clip((cols-X0)/(X1-X0), 0, 1)-.5         # -0.5 .. +0.5 across the window
    ink = np.asarray(zarr.open(f'{args.labels}/pherc0139-w043_inklabels.zarr/0', mode='r')[10]) > 0
    sup = np.asarray(zarr.open(f'{args.labels}/pherc0139-w043_supervision_mask.zarr/0', mode='r')[10]) > 0
    rows = []
    for tilt in [float(t) for t in args.tilts_um.split(',')]:
        name = f'tilt{int(tilt):03d}'
        mesh = work/f'mesh-{name}'
        mesh.mkdir(exist_ok=True)
        shifted = pts+n*(ramp[None, :, None]*tilt/voxel)
        for i, a in enumerate('zyx'):
            arr = np.where(valid, shifted[..., i], -1).astype(np.float32)
            tifffile.imwrite(mesh/f'{a}.tif', arr)
        (mesh/'meta.json').write_text((work/'mesh'/'meta.json').read_text())
        sv = work/f'w043-{name}.zarr'
        if not (sv/'render.json').exists():
            subprocess.run([sys.executable, str(HERE/'render_tifxyz_light.py'), str(mesh), VOLUME, str(sv),
                            '--seg-level-factor', '1', '--depth', '65', '--crop', str(Y0), str(Y1), str(X0), str(X1)],
                           check=True, stdout=subprocess.DEVNULL)
        qc = profile(str(sv), voxel)
        out = work/f'w043-{name}-reverse.tif'
        if not out.exists():
            subprocess.run([sys.executable, str(HERE/'canonical2um_infer.py'), str(sv), args.checkpoint, str(out),
                            '--start', '1', '--end', '63', '--stride', '128', '--reverse'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p = tifffile.imread(out).astype(np.float32)/255
        p4 = p.reshape(p.shape[0]//4, 4, p.shape[1]//4, 4).mean((1, 3))
        i = ink[Y0//4:Y0//4+p4.shape[0], X0//4:X0//4+p4.shape[1]]
        s = sup[Y0//4:Y0//4+p4.shape[0], X0//4:X0//4+p4.shape[1]]
        row = {'tilt_um': tilt, 'S': qc['S'], 'S_local_median': qc['S_local_median'],
               'peak_offset_sd_um': qc['peak_offset_sd_um'], 'auc': float(roc_auc_score(i[s], p4[s]))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (work/'tilt_experiment.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
