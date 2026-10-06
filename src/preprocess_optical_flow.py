import numpy as np
import os
import sys

# Anchor imports to AioDeflicker root regardless of CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AIO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _AIO_ROOT)
# set gpu
from pathlib import Path
import argparse
from src.models.stage_1.raft_wrapper import RAFTWrapper

from tqdm import tqdm

def preprocess(args):
    files = sorted(args.vid_path.glob('*.*g'))
    
    if args.ignore_start > 0 or args.ignore_end > 0:
        end_idx = len(files) - args.ignore_end
        files = files[args.ignore_start : end_idx]
        print(f"[preprocess_optical_flow] Processing range {args.ignore_start} to {end_idx - 1} (total processed: {len(files)})")

    vid_name = args.vid_path.name
    vid_root = args.vid_path.parent
    out_flow_dir = vid_root / f'{vid_name}_flow'
    out_flow_dir.mkdir(exist_ok=True)
    raft_model_path = os.path.join(args.weights_dir, 'raft-things.pth')
    raft_wrapper = RAFTWrapper(
        model_path=raft_model_path, max_long_edge=args.max_long_edge)
    for i, file1 in enumerate(tqdm(files, desc='computing flow', miniters=10)):
        if i < len(files) - 1:
            file2 = files[i + 1]
            fn1 = file1.name
            fn2 = file2.name
            out_flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
            out_flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'

            overwrite=False
            if not out_flow12_fn.exists() and not out_flow21_fn.exists() or overwrite:
                im1, im2 = raft_wrapper.load_images(str(file1), str(file2))
                flow12 = raft_wrapper.compute_flow(im1, im2)
                flow21 = raft_wrapper.compute_flow(im2, im1)
                np.save(out_flow12_fn, flow12)
                np.save(out_flow21_fn, flow21)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument(
        '--vid-path', type=Path, default=Path('./data/'), help='folder to process')
    # Need to reduce when there are memory issues
    parser.add_argument('--max_long_edge', type=int,default='2000', help='maximum image dimension to process without resizing')
    parser.add_argument('--gpu', type=int,default=0, help='gpu id')
    parser.add_argument('--weights_dir', type=str,
                        default=os.path.join(_AIO_ROOT, 'pretrained_weights'),
                        help='directory containing pretrained weight files')
    parser.add_argument('--ignore_start', type=int, default=0)
    parser.add_argument('--ignore_end', type=int, default=0)

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    preprocess(args=args)
