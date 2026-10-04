"""Pick seeds on a team surface prediction and grow a patch with vc_grow_seg_from_seed (VC3D Docker image).

Works on any scan whose surface prediction and normal grids are published in the open-data bucket,
including predictions made at a pyramid level other than 0 (e.g. the m7 "L2" predictions of the
2.4 µm scans): seeds and growth then live in that level's voxel coordinates.

Grid prefetching, seed placement and GrowPatch parameters are adapted from Bullo27/first-letters-survey
(fls.py, https://github.com/Bullo27/first-letters-survey), used under its MIT licence:
    Copyright (c) 2026 Matteo Bulloni. Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software ... (full MIT text in LICENSE-THIRD-PARTY).

Example (PHerc0846A, 2.403 µm scan, L2 prediction):

    python grow_patch.py seeds --work ~/fl-work
    python grow_patch.py grow --work ~/fl-work --seed-index 12 --gens 100

IMPORTANT: keep --work on a native Linux filesystem (e.g. inside WSL ext4). On a Windows drive mounted
through Docker Desktop the tracer is I/O bound on grid reads (we measured 50 generations in 3 h there,
versus 15 generations in 22 s and 100 generations in 22 min on ext4).
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import ssl
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

HOST = 'vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
BUCKET = f'https://{HOST}/'
DEFAULTS = {
    'scan': 'PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr',
    'pred': 'PHerc0846A/representations/predictions/surfaces/'
            '20260319102732-surface-20260413222639-surface-m7-L2-th0.2.zarr',
    'grids': 'PHerc0846A/representations/predictions/surfaces/'
             '20260319102732-surface-20260413222639-surface-m7-L2-th0.2.normal-grids',
    'pred_level': 2,
    'scan_voxel_um': 2.403,
}
IMAGE = 'ghcr.io/scrollprize/villa/volume-cartographer:edge'
_tls = threading.local()


def log(message):
    print(time.strftime('%H:%M:%S ')+message, flush=True)


def bucket_get(key, tries=6):
    for attempt in range(tries):
        conn = getattr(_tls, 'conn', None)
        if conn is None:
            conn = _tls.conn = http.client.HTTPSConnection(HOST, timeout=60, context=ssl.create_default_context())
        try:
            conn.request('GET', '/'+key)
            response = conn.getresponse()
            body = response.read()
            if response.status == 200:
                return body
            if response.status == 404:
                return None
            raise IOError(f'HTTP {response.status}')
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _tls.conn = None
            time.sleep(.5+attempt)
    raise IOError(f'giving up on {key}')


def pred_shape(pred):
    return json.loads(bucket_get(f'{pred}/0/.zarray'))['shape']


def seeds(args, zfs=(0.3, 0.5, 0.7), rfs=(0.5, 0.7), angles=(0.8, 3.0, 5.2)):
    """Seeds relative to the papyrus: scan level 5 mask and the prediction level that shares its grid."""
    import zarr
    so = {'anon': True}
    match = 5-args.pred_level           # prediction level with the same voxel pitch as scan level 5
    m5 = zarr.open(f's3://vesuvius-challenge-open-data/{args.scan}/5', mode='r', storage_options=so)
    pm = zarr.open(f's3://vesuvius-challenge-open-data/{args.pred}/{match}', mode='r', storage_options=so)
    p0 = zarr.open(f's3://vesuvius-challenge-open-data/{args.pred}/0', mode='r', storage_options=so)
    if m5.shape != pm.shape:
        raise SystemExit(f'scan L5 {m5.shape} and prediction L{match} {pm.shape} do not align')
    step = 2**match
    out = []
    for zf in zfs:
        z5 = int(m5.shape[0]*zf)
        mask = np.asarray(m5[max(0, z5-1):z5+2]).max(axis=0) > 0
        ys, xs = np.nonzero(mask)
        if len(ys) < 30:
            continue
        cy, cx = ys.mean(), xs.mean()
        rmask = np.percentile(np.hypot(ys-cy, xs-cx), 99)
        lit = np.asarray(pm[max(0, z5-1):z5+2]).max(axis=0) > 127
        py, px = np.nonzero(lit & mask)        # the prediction also fires outside the papyrus
        if not len(py):
            continue
        for rf in rfs:
            for angle in angles:
                ty, tx = cy+rf*rmask*np.sin(angle), cx+rf*rmask*np.cos(angle)
                i = int(np.argmin((py-ty)**2+(px-tx)**2))
                if (py[i]-ty)**2+(px[i]-tx)**2 > (0.08*rmask)**2:
                    continue
                Z, Y, X, h = z5*step, int(py[i])*step, int(px[i])*step, 2*step
                oz, oy, ox = max(0, Z-h), max(0, Y-h), max(0, X-h)
                box = np.asarray(p0[oz:Z+h, oy:Y+h, ox:X+h])
                zz, yy, xx = np.nonzero(box > 127)
                if not len(zz):
                    continue
                j = int(np.argmin((zz+oz-Z)**2+(yy+oy-Y)**2+(xx+ox-X)**2))
                out.append({'zf': zf, 'rf': rf, 'angle': angle,
                            'x': int(ox+xx[j]), 'y': int(oy+yy[j]), 'z': int(oz+zz[j])})
    work = Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    (work/'seeds.json').write_text(json.dumps(out, indent=1))
    for k, s in enumerate(out):
        print(k, json.dumps(s))


def fetch_grids(args, x, y, z, workers=48):
    d = Path(args.work).expanduser()/'normal_grids'
    d.mkdir(parents=True, exist_ok=True)
    if not (d/'normal-grids-remote.json').exists():
        (d/'normal-grids-remote.json').write_text(json.dumps({'url': BUCKET+args.grids}))
    if not (d/'metadata.json').exists():
        (d/'metadata.json').write_bytes(bucket_get(args.grids+'/metadata.json'))
    sp = max(1, int(json.loads((d/'metadata.json').read_text()).get('sparse-volume', 1)))
    reach = int(args.gens*20*1.15)
    zs, ys, xs = pred_shape(args.pred)

    def span(c, n):
        return range(max(0, (c-reach)//sp*sp), min(n-1, c+reach)+1, sp)
    jobs = [('xy', i) for i in span(z, zs)]+[('xz', i) for i in span(y, ys)]+[('yz', i) for i in span(x, xs)]

    def one(job):
        rel = f'{job[0]}/{job[1]:06d}.grid'
        dst = d/rel
        if dst.exists() or Path(str(dst)+'.missing').exists():
            return 'have'
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = bucket_get(f'{args.grids}/{rel}')
        except IOError:
            return 'fail'
        if body is None:
            Path(str(dst)+'.missing').touch()
            return '404'
        tmp = Path(f'{dst}.part{os.getpid()}_{threading.get_ident()}')
        tmp.write_bytes(body)
        os.replace(tmp, dst)
        return 'ok'
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, jobs))
    log(f'grids: {len(jobs)} slices, fetched {res.count("ok")}, cached {res.count("have")}, failed {res.count("fail")}')


def grow(args):
    work = Path(args.work).expanduser()
    s = json.loads((work/'seeds.json').read_text())[args.seed_index]
    name = f"s{args.seed_index}-{s['x']}_{s['y']}_{s['z']}-g{args.gens}"
    out = work/'patches'/name
    out.mkdir(parents=True, exist_ok=True)
    fetch_grids(args, s['x'], s['y'], s['z'])
    params = {'mode': 'seed', 'step_size': 20, 'generations': args.gens, 'min_area_cm': 0,
              'thread_limit': args.threads, 'normal_grid_path': '/work/normal_grids',
              'voxelsize': args.scan_voxel_um*2**args.pred_level, 'cache_size': 4_000_000_000,
              'cache_root': '/work/tracer_cache'}
    (out/'params.json').write_text(json.dumps(params, indent=1))
    started = time.time()
    command = ['docker', 'run', '--rm', '-e', 'OPENBLAS_NUM_THREADS=1', '-v', f'{work}:/work', args.image,
               'vc_grow_seg_from_seed', BUCKET+args.pred, f'/work/patches/{name}',
               f'/work/patches/{name}/params.json', str(s['x']), str(s['y']), str(s['z'])]
    with open(out/'grow.log', 'a') as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    segs = sorted(p for p in out.iterdir() if p.name.startswith('auto_grown_'))
    if rc or not segs:
        raise SystemExit(f'grow failed rc={rc}; see {out}/grow.log')
    meta = json.loads((segs[-1]/'meta.json').read_text())
    area_cm2 = meta['area_vx2']*(args.scan_voxel_um*2**args.pred_level*1e-4)**2
    log(f'grow: {area_cm2:.2f} cm² in {time.time()-started:.0f} s -> {segs[-1]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cmd', choices=('seeds', 'grow'))
    parser.add_argument('--work', required=True)
    parser.add_argument('--scan', default=DEFAULTS['scan'])
    parser.add_argument('--pred', default=DEFAULTS['pred'])
    parser.add_argument('--grids', default=DEFAULTS['grids'])
    parser.add_argument('--pred-level', type=int, default=DEFAULTS['pred_level'])
    parser.add_argument('--scan-voxel-um', type=float, default=DEFAULTS['scan_voxel_um'])
    parser.add_argument('--seed-index', type=int)
    parser.add_argument('--gens', type=int, default=100)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--image', default=IMAGE)
    args = parser.parse_args()
    seeds(args) if args.cmd == 'seeds' else grow(args)


if __name__ == '__main__':
    main()
