import os
import sys
import argparse
import subprocess
import shutil

# ── args ────────────────────────────────────────────────────────────────────
aio_root = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(aio_root, 'pretrained_weights')

parser = argparse.ArgumentParser(description='AIO-Deflicker: video deflickering pipeline')
parser.add_argument('--input_video',          type=str,   required=True,  help='Input video path (mp4)')
parser.add_argument('--output_video',         type=str,   required=True,  help='Output video path (mp4)')
parser.add_argument('--ckpt_filter',          type=str,   default=os.path.join(weights_dir, 'neural_filter.pth'),
                    help='Checkpoint for neural filter network')
parser.add_argument('--ckpt_local',           type=str,   default=os.path.join(weights_dir, 'local_refinement_net.pth'),
                    help='Checkpoint for local refinement network')
parser.add_argument('--gpu',                  type=int,   default=0,      help='GPU device id')
parser.add_argument('--class_name',           type=str,   default=None,   help='Class for segmentation (None = disabled)')
parser.add_argument('--down',                 type=int,   default=4,      help='Downsample rate')
parser.add_argument('--iters',                type=int,   default=10000,  help='Atlas training iterations')
parser.add_argument('--optical_flow_coeff',   type=float, default=None,   help='Optical flow coefficient')
parser.add_argument('--alpha_flow_factor',    type=float, default=None,   help='Alpha flow factor')
parser.add_argument('--ignore_start',         type=int,   default=0,      help='Number of frames to ignore at the start')
parser.add_argument('--ignore_end',           type=int,   default=0,      help='Number of frames to ignore at the end')

opts = parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def get_video_fps(video_path):
    cmd = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=r_frame_rate,avg_frame_rate '
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    )
    try:
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        lines = [line.strip() for line in output.splitlines() if line.strip() and line.strip() != '0/0']
        for rate in lines:
            if '/' in rate:
                num, den = map(int, rate.split('/'))
                if den != 0:
                    val = num / den
                    if abs(val - round(val)) < 0.005:
                        return f"{int(round(val))}/1"
                    for n, d in [(24000, 1001), (30000, 1001), (48000, 1001), (60000, 1001), (120000, 1001), (240000, 1001)]:
                        if abs(val - n / d) < 0.005:
                            return f"{n}/{d}"
                    return f"{num}/{den}"
            elif rate:
                val = float(rate)
                if abs(val - round(val)) < 0.005:
                    return f"{int(round(val))}/1"
                for n, d in [(24000, 1001), (30000, 1001), (48000, 1001), (60000, 1001), (120000, 1001), (240000, 1001)]:
                    if abs(val - n / d) < 0.005:
                        return f"{n}/{d}"
                return rate
    except Exception as e:
        print(f"Warning: Could not get FPS via ffprobe: {e}")
    raise RuntimeError(f"Could not determine valid FPS for '{video_path}'")


def get_video_size(video_path):
    cmd = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=width,height '
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    )
    try:
        result = subprocess.check_output(cmd, shell=True).decode().strip().split()
        if len(result) >= 2:
            return int(result[0]), int(result[1])
    except Exception:
        pass
    return None, None


def run(cmd, env=None, cwd=None):
    """Run a shell command and raise on error."""
    print('>', cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd,
                   shell=isinstance(cmd, str))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import math
    input_video  = os.path.abspath(opts.input_video)
    output_video = os.path.abspath(opts.output_video)

    fps = get_video_fps(input_video)
    print(f'[aiodeflicker] FPS: {fps}')
    print(f'[aiodeflicker] Input : {input_video}')
    print(f'[aiodeflicker] Output: {output_video}')

    w_orig, h_orig = get_video_size(input_video)
    if w_orig is not None and h_orig is not None:
        w_target = int(math.ceil(w_orig / 32.0) * 32)
        h_target = int(math.ceil(h_orig / 32.0) * 32)
        print(f'[aiodeflicker] Original resolution: {w_orig}x{h_orig}')
        print(f'[aiodeflicker] Divisible-by-32 target resolution: {w_target}x{h_target}')
        if w_target != w_orig or h_target != h_orig:
            print(f'[aiodeflicker] Resizing input to {w_target}x{h_target} and back to {w_orig}x{h_orig} in final output.')
            use_resize = True
        else:
            use_resize = False
    else:
        print('[aiodeflicker] Warning: Could not determine video dimensions. Running without preprocessing resize.')
        use_resize = False
        w_target, h_target = None, None

    import uuid
    vid_name = f"run_{uuid.uuid4().hex[:8]}"
    
    # Create work directory next to the output video
    output_dir = os.path.dirname(output_video)
    work_dir = os.path.join(output_dir, 'aiodeflicker', vid_name)
    os.makedirs(work_dir, exist_ok=True)
    
    # Directories for intermediate data — all inside work_dir
    data_root   = work_dir           # sub-scripts will look for <data_root>/<vid_name>/
    frames_dir  = os.path.join(work_dir, vid_name)   # actual frames go here
    results_dir = os.path.join(work_dir, 'results')   # sub-scripts write results here
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Absolute paths for sub-scripts in AioDeflicker/src/
    src_dir = os.path.join(aio_root, 'src')

    try:

        # ── 1. Extract input frames ──────────────────────────────────────────
        print('[aiodeflicker] Extracting frames...')
        if use_resize:
            extract_vf = f'fps={fps},scale={w_target}:{h_target}:flags=area'
        else:
            extract_vf = f'fps={fps}'
        run(
            f'ffmpeg -y -i "{input_video}" -vf "{extract_vf}" -start_number 0 '
            f'"{frames_dir}/%05d.png"'
        )

        # Env with PYTHONPATH so sub-scripts can import src.*
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        env['PYTHONPATH'] = aio_root + os.pathsep + env.get('PYTHONPATH', '')

        # ── 2. Compute optical flow ──────────────────────────────────────────
        print('[aiodeflicker] Computing optical flow...')
        flow_cmd = [
            sys.executable, os.path.join(src_dir, 'preprocess_optical_flow.py'),
            '--vid-path', frames_dir,
            '--gpu', str(opts.gpu),
            '--weights_dir', weights_dir,
            '--ignore_start', str(opts.ignore_start),
            '--ignore_end', str(opts.ignore_end),
        ]
        run(flow_cmd, env=env)

        # ── 3. Atlas training ────────────────────────────────────────────────
        print('[aiodeflicker] Training neural atlas...')
        atlas_script = os.path.join(src_dir,
            'stage1_neural_atlas_seg.py' if opts.class_name is not None
            else 'stage1_neural_atlas.py')
        atlas_cmd = [
            sys.executable, atlas_script,
            '--vid_name',     vid_name,
            '--root',         data_root,
            '--down',         str(opts.down),
            '--iters',        str(opts.iters),
            '--gpu',          str(opts.gpu),
            '--results_root', results_dir,
            '--weights_dir',  weights_dir,
        ]
        if opts.class_name is not None:
            atlas_cmd += ['--class_name', opts.class_name]
        if opts.optical_flow_coeff is not None:
            atlas_cmd += ['--optical_flow_coeff', str(opts.optical_flow_coeff)]
        if opts.alpha_flow_factor is not None:
            atlas_cmd += ['--alpha_flow_factor', str(opts.alpha_flow_factor)]
        
        atlas_cmd += ['--ignore_start', str(opts.ignore_start), '--ignore_end', str(opts.ignore_end)]
        run(atlas_cmd, env=env)

        # ── 4. Neural filter + local refinement ──────────────────────────────
        print('[aiodeflicker] Running neural filter and local refinement...')
        refine_cmd = [
            sys.executable, os.path.join(src_dir, 'neural_filter_and_refinement.py'),
            '--video_name',   vid_name,
            '--fps',          str(fps),
            '--ckpt_filter',  opts.ckpt_filter,
            '--ckpt_local',   opts.ckpt_local,
            '--gpu',          str(opts.gpu),
            '--data_root',    data_root,
            '--results_root', results_dir,
            '--ignore_start', str(opts.ignore_start),
            '--ignore_end',   str(opts.ignore_end),
        ]
        run(refine_cmd, env=env)

        # ── 5. Assemble output video from final frames ────────────────────────
        # neural_filter_and_refinement.py writes final frames to:
        #   <results_dir>/<vid_name>/final/output/%05d.png
        final_frames_dir = os.path.join(
            results_dir, vid_name, 'final', 'output')

        print(f'[aiodeflicker] Encoding output video from {final_frames_dir}...')
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        if use_resize:
            encode_vf = f' -vf "scale={w_orig}:{h_orig}:flags=area"'
        else:
            encode_vf = ''
        run(
            f'ffmpeg -y -r {fps} -i "{final_frames_dir}/%05d.png"{encode_vf} '
            f'-c:v libx264 -crf 16 -preset slower -pix_fmt yuv420p "{output_video}"'
        )

        
        print(f'[aiodeflicker] Done Encoding -> {output_video}')

    finally:
        # ── 6. Clean up intermediate files ────────────────────────────────────
        print(f'[aiodeflicker] Cleaning up intermediate files in {work_dir}...')
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            
        # Also try removing aiodeflicker folder if empty
        aiodeflicker_parent = os.path.dirname(work_dir)
        try: os.rmdir(aiodeflicker_parent)
        except: pass
        
        print(f'[aiodeflicker] Done cleanup.')


if __name__ == '__main__':
    main()
