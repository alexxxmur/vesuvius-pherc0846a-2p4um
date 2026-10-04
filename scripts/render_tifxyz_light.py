"""Lightweight tifxyz → surface-volume renderer that fetches only the level-0 chunks the shell touches.

vc_render_tifxyz in the edge image crashed and pulled ~31 GB for a 0.5 cm² patch on the 2.4 µm
PHerc0846A scan (FL2). Here the tifxyz grid (given in level-`--seg-level` voxels) is upsampled to
level-0 pixel density, normals are taken from the grid, and each output tile samples ±half-depth
level-0 voxels along the normal with trilinear interpolation from cached raw chunks.
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import ssl
import threading
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


HOST = 'vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
_tls = threading.local()


def fetch(key, tries=6):
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
            if response.status in (403, 404):
                return None
            raise IOError(response.status)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _tls.conn = None
            time.sleep(.5+attempt)
    raise IOError(key)


class ChunkStore:
    def __init__(self, volume_key, cache_dir):
        meta = json.loads(fetch(f'{volume_key}/0/.zarray'))
        if meta['compressor'] is not None or meta['dtype'] != '|u1':
            raise ValueError(f'expected raw uint8 chunks: {meta}')
        self.key, self.shape, self.chunk = volume_key, meta['shape'], meta['chunks']
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fetched_bytes = 0
        self.lock = threading.Lock()

    def get(self, cz, cy, cx):
        path = self.cache/f'{cz}.{cy}.{cx}'
        raw = None
        if path.exists():
            try:
                raw = path.read_bytes()
            except OSError:
                # On /mnt/d (9p) a file another worker is renaming can be listed but not yet readable.
                raw = None
        if raw is None:
            raw = fetch(f'{self.key}/0/{cz}/{cy}/{cx}') or b''
            tmp = path.with_suffix(f'.part{os.getpid()}_{threading.get_ident()}')
            try:
                tmp.write_bytes(raw)
                os.replace(tmp, path)
            except OSError:
                tmp.unlink(missing_ok=True)   # another worker cached the same chunk; ours is in memory
            with self.lock:
                self.fetched_bytes += len(raw)
        if not raw:
            return np.zeros(self.chunk, np.uint8)
        return np.frombuffer(raw, np.uint8).reshape(self.chunk)

    def block(self, lo, hi, workers=16):
        """Dense uint8 block covering voxel box [lo, hi) (z, y, x), assembled from chunks."""
        c = self.chunk
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.shape)
        if (hi <= lo).any():
            # The shell lies entirely outside the scan here; sampling then returns the fill value 0.
            return np.zeros((1, 1, 1), np.uint8), lo, 0
        ranges = [range(lo[i]//c[i], (hi[i]-1)//c[i]+1) for i in range(3)]
        out = np.zeros(hi-lo, np.uint8)
        keys = [(z, y, x) for z in ranges[0] for y in ranges[1] for x in ranges[2]]
        with cf.ThreadPoolExecutor(workers) as ex:
            for (z, y, x), data in zip(keys, ex.map(lambda k: self.get(*k), keys)):
                z0, y0, x0 = z*c[0], y*c[1], x*c[2]
                sz = slice(max(z0, lo[0]), min(z0+c[0], hi[0]))
                sy = slice(max(y0, lo[1]), min(y0+c[1], hi[1]))
                sx = slice(max(x0, lo[2]), min(x0+c[2], hi[2]))
                out[sz.start-lo[0]:sz.stop-lo[0], sy.start-lo[1]:sy.stop-lo[1], sx.start-lo[2]:sx.stop-lo[2]] = \
                    data[sz.start-z0:sz.stop-z0, sy.start-y0:sy.stop-y0, sx.start-x0:sx.stop-x0]
        return out, lo, len(keys)


def load_grid(seg):
    import tifffile
    x, y, z = (tifffile.imread(Path(seg)/f'{a}.tif').astype(np.float64) for a in 'xyz')
    valid = (x > 0) & (y > 0) & (z > 0)
    meta = json.loads((Path(seg)/'meta.json').read_text())
    return np.stack([z, y, x], -1), valid, meta['scale'][0]


def fill_invalid(points, valid):
    """Nearest-valid fill so upsampling and derivatives stay finite; invalid output stays masked."""
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return points[idx[0], idx[1]]


def normals(points, center_yx):
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    n = np.cross(dv, du)
    for c in range(3):
        n[..., c] = gaussian_filter(n[..., c], 1.0)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)+1e-9
    # The grid's own normals are consistently oriented across the patch. Choose ONE sign for the whole
    # patch (majority vote: away from the image centre) instead of per node: a per-node sign flipped the
    # depth order on up to half of a patch wherever the sheet runs roughly radially.
    radial = points[..., 1:]-np.asarray(center_yx)
    radial /= np.linalg.norm(radial, axis=-1, keepdims=True)+1e-9
    vote = np.sum(n[..., 1:]*radial, axis=-1)
    return n*(1.0 if vote.sum() >= 0 else -1.0)


_JOB = None


def render_tile(tile_origin):
    """Worker: sample one output tile along the normals and write it into the level-0 array."""
    import zarr
    job = _JOB
    _tls.conn = None          # never reuse an HTTPS connection inherited across fork
    ty, tx = tile_origin
    th, tw = min(job['tile'], job['h']-ty), min(job['tile'], job['w']-tx)
    store = ChunkStore(job['volume_key'], job['cache'])
    points, n = job['points'], job['n']
    # Pixel p of the full canvas sits on grid coordinate p*scale (VC convention: node i at pixel i/scale).
    gs = (points.shape[0]/job['H'], points.shape[1]/job['W'])
    gy = np.arange(ty, ty+th)*gs[0]
    gx = np.arange(tx, tx+tw)*gs[1]
    GY, GX = np.meshgrid(gy, gx, indexing='ij')
    p = np.stack([map_coordinates(points[..., c], [GY, GX], order=1, mode='nearest') for c in range(3)], -1)
    nn = np.stack([map_coordinates(n[..., c], [GY, GX], order=1, mode='nearest') for c in range(3)], -1)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)+1e-9
    coords = p[None]+job['offsets'][:, None, None, None]*nn[None]
    lo = np.floor(coords.reshape(-1, 3).min(0)).astype(int)-1
    hi = np.ceil(coords.reshape(-1, 3).max(0)).astype(int)+2
    block, lo, touched = store.block(lo, hi, workers=8)
    values = map_coordinates(block.astype(np.float32), (coords-lo).reshape(-1, 3).T, order=1, mode='constant', cval=0)
    out = np.clip(values, 0, 255).reshape(job['depth'], th, tw).astype(np.uint8)
    out[:, ~job['mask'][ty:ty+th, tx:tx+tw]] = 0
    oy, ox = job['origin']
    zarr.open(str(Path(job['output'])/'0'), mode='r+')[:, ty-oy:ty-oy+th, tx-ox:tx-ox+tw] = out
    return touched, store.fetched_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segment')
    parser.add_argument('volume_key', help='bucket key of the OME-Zarr scan root')
    parser.add_argument('output')
    parser.add_argument('--seg-level-factor', type=float, default=4.0, help='level-0 voxels per segment voxel')
    parser.add_argument('--depth', type=int, default=65)
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--cache', default=str(Path.home()/'.cache/vc-chunks'), help='raw chunk cache (put it on a big disk)')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('Y0', 'Y1', 'X0', 'X1'),
                        help='render only this output-pixel window (multiples of --tile)')
    args = parser.parse_args()
    import zarr
    store = ChunkStore(args.volume_key, Path(args.cache)/args.volume_key.replace('/', '_'))
    points, valid, scale = load_grid(args.segment)
    points = fill_invalid(points, valid)*args.seg_level_factor          # level-0 voxels
    center_yx = (store.shape[1]/2, store.shape[2]/2)
    n = normals(points, center_yx)
    up = 1/scale*args.seg_level_factor                                    # output px per grid step
    h, w = int(round(points.shape[0]*up)), int(round(points.shape[1]*up))
    half = args.depth//2
    # Same layout as the published surface volumes: a group whose level-0 array is "0".
    root = zarr.open_group(args.output, mode='w', zarr_format=2)
    y0, y1, x0, x1 = args.crop if args.crop else (0, h, 0, w)
    root.create_array('0', shape=(args.depth, y1-y0, x1-x0), chunks=(args.depth, 128, 128), dtype='u1',
                      compressors=None, fill_value=0)
    mask_up = zoom(valid.astype(np.float32), (h/valid.shape[0], w/valid.shape[1]), order=1) > .5
    mask_up[:y0] = False
    mask_up[y1:] = False
    mask_up[:, :x0] = False
    mask_up[:, x1:] = False
    tiles = [(ty, tx) for ty in range(y0, y1, args.tile) for tx in range(x0, x1, args.tile)
             if mask_up[ty:ty+args.tile, tx:tx+args.tile].any()]
    global _JOB
    _JOB = {'points': points, 'n': n, 'mask': mask_up, 'h': min(h, y1), 'w': min(w, x1), 'H': h, 'W': w,
            'tile': args.tile,
            'origin': (y0, x0),
            'offsets': np.arange(-half, args.depth-half, dtype=np.float32), 'depth': args.depth,
            'output': args.output, 'volume_key': args.volume_key,
            'cache': str(Path(args.cache)/args.volume_key.replace('/', '_'))}
    started, chunks_touched, fetched, done = time.time(), 0, 0, 0
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context('fork')) as ex:
        for touched, got in ex.map(render_tile, tiles):
            chunks_touched += touched
            fetched += got
            done += 1
            if done % 20 == 0 or done == len(tiles):
                print(json.dumps({'tiles': done, 'of': len(tiles), 'chunks_touched': chunks_touched,
                                  'fetched_gb': round(fetched/1e9, 2), 'seconds': round(time.time()-started)}),
                      flush=True)
    Path(args.output, 'render.json').write_text(json.dumps({
        'segment': str(args.segment), 'volume_key': args.volume_key, 'depth': args.depth,
        'seg_level_factor': args.seg_level_factor, 'shape': [args.depth, y1-y0, x1-x0],
        'full_canvas': [h, w], 'crop': [y0, y1, x0, x1],
        'offset_order': 'index 0 = most inward (toward scroll axis); this is the REVERSE of vc_render_tifxyz order on our w043 check', 'fetched_gb': fetched/1e9,
        'seconds': time.time()-started}, indent=1))


if __name__ == '__main__':
    main()
