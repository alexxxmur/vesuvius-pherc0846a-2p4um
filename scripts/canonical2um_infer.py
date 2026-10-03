"""Run the canonical 2 µm ink model (scrollprize/ink_canonical_2um, ResNet-152 + 3D decoder) on a local
surface volume, using villa's ink-detection/optimized_inference code unchanged.

    python canonical2um_infer.py surface.zarr r152_3ddec_v2_l5_epoch13.ckpt out.tif --start 33 --end 95 --reverse

--villa-opt (or $VILLA_INK_OPT) must point to villa/ink-detection/optimized_inference.
The model expects a 62-slice window centred on the papyrus surface and 256-px tiles.
Layer order matters: on our w043 check the same window gave AUC 0.87 in one order and 0.44 in the
other, so always run both (--reverse).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('volume', help='surface-volume zarr (group with "0", depth first)')
    parser.add_argument('checkpoint')
    parser.add_argument('output_tif')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=63)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--villa-opt', default=os.environ.get('VILLA_INK_OPT'))
    args = parser.parse_args()
    if not args.villa_opt:
        raise SystemExit('set --villa-opt or VILLA_INK_OPT to villa/ink-detection/optimized_inference')
    sys.path.insert(0, args.villa_opt)
    import tifffile
    import torch
    import zarr
    from inference import CFG, run_inference
    from model_resnet3d_3d_decoder import load_model
    CFG.in_chans = args.end-args.start
    CFG.tile_size = CFG.size = 256
    CFG.stride = args.stride
    CFG.batch_size = args.batch_size
    CFG.num_parts, CFG.part_id = 1, 0
    device = torch.device('cuda')
    model = load_model(args.checkpoint, device, num_frames=CFG.in_chans)
    with tempfile.TemporaryDirectory(dir=str(Path(args.output_tif).parent)) as tmp:
        CFG.zarr_output_dir = tmp
        result = run_inference(args.volume, model, device, is_reverse_segment=args.reverse,
                               start_z=args.start, end_z=args.end)
        pred = np.asarray(zarr.open(result['mask_pred'], mode='r')[:])
        count = np.asarray(zarr.open(result['mask_count'], mode='r')[:])
    prob = np.where(count > 0, pred/np.maximum(count, 1e-6), 0)
    tifffile.imwrite(args.output_tif, (np.clip(prob, 0, 1)*255).astype(np.uint8), compression='lzw')


if __name__ == '__main__':
    main()
