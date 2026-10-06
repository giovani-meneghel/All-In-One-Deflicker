import os
import sys

# Anchor imports to AioDeflicker root regardless of CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AIO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _AIO_ROOT)
import numpy as np
from glob import glob
import shutil
import src.models.network_filter as net
from src.models.utils import tensor2img, load_image, InputPadder
import argparse
import random
import torch
from tqdm import tqdm 
import cv2

### custom lib
from src.models.network_local import TransformNet
import src.models.utils as utils
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_filter", default=os.path.join(_AIO_ROOT, 'pretrained_weights', 'neural_filter.pth'),
                    type=str, help="the ckpt of neural filter network")
parser.add_argument("--ckpt_local", default=os.path.join(_AIO_ROOT, 'pretrained_weights', 'local_refinement_net.pth'),
                    type=str, help="the ckpt of local refinement network")
parser.add_argument("--fps", default=10, type=float, help="frame per second")
parser.add_argument("--video_name", default=None, type=str, help="the name of input video")
parser.add_argument('--gpu',             type=int,     default=0,                help='gpu device id')
parser.add_argument('--ignore_start',    type=int,     default=0,                help='ignore frames at start')
parser.add_argument('--ignore_end',      type=int,     default=0,                help='ignore frames at end')
parser.add_argument('--data_root',       type=str,     default=os.path.join(_AIO_ROOT, 'data', 'test'),
                    help='Root directory containing input frame folders')
parser.add_argument('--results_root',    type=str,     default=os.path.join(_AIO_ROOT, 'results'),
                    help='Root directory for results output')

# set random seed
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# process arguments
opts = parser.parse_args()
print(opts)

# set gpu

if not torch.cuda.is_available():
    raise Exception("No GPU found, run with cpu")
    device = torch.device("cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda:{}".format(opts.gpu))

# Define neural filter model
filter_net = net.UNet(in_channels=6, out_channels=3, init_features=32)

# load ckpt
ckpt = torch.load(opts.ckpt_filter)
filter_net.load_state_dict(ckpt)
filter_net.to(device)
filter_net.eval()

# ## Define local refinement model
# ## Local refinement net is rely on Lai et al., ECCV 2018, thank you!

### initialize model
model_opts = edict({'nf':32, 'norm':'IN', 'model':'TransformNet', 'blocks': 5})
local_net = TransformNet(model_opts, nc_in=12, nc_out=3)

### load trained model
print("Load %s" % opts.ckpt_local)
ckpt_local = torch.load(opts.ckpt_local) 
local_net.load_state_dict(ckpt_local)
local_net = local_net.to(device)
local_net.eval()


style_root = os.path.join(opts.results_root, opts.video_name, "stage_1", "output")
content_root = os.path.join(opts.data_root, opts.video_name)
style_names = sorted(glob(style_root + "/*"))
content_names = sorted(glob(content_root + "/*"))
assert len(style_names) == len(content_names), "the number of style frames is different from the number of content frames"
num_frames = len(style_names)
print("Processing {} frames".format(num_frames))

# setup folder
output_folder = os.path.join(opts.results_root, opts.video_name, "neural_filter", "concat")
os.makedirs(output_folder, exist_ok=True)
process_filter_dir = os.path.join(opts.results_root, opts.video_name, "neural_filter", "output")
os.makedirs(process_filter_dir, exist_ok=True)
output_final_dir = os.path.join(opts.results_root, opts.video_name, "final", "output")
os.makedirs(output_final_dir, exist_ok=True)

print("neural filter dir:", process_filter_dir)
print("output final dir:", output_final_dir)
print("output dir:", output_folder) # concat output

first_processed_id = opts.ignore_start
last_processed_id = num_frames - opts.ignore_end - 1

for frame_id in tqdm(range(num_frames), miniters=10):
    if frame_id < opts.ignore_start or frame_id > last_processed_id:
        ### bypass processing
        shutil.copy(content_names[frame_id], "{}/{:05d}.png".format(output_folder, frame_id))
        shutil.copy(content_names[frame_id], "{}/{:05d}.png".format(process_filter_dir, frame_id))
        shutil.copy(content_names[frame_id], "{}/{:05d}.png".format(output_final_dir, frame_id))
        continue

    ### neural filter net
    frame_content, org_size = load_image(content_names[frame_id], device=device, resize=False)
    frame_style, _ = load_image(style_names[frame_id], size=org_size, device=device, resize=False)
    padder = InputPadder(frame_content.shape)
    frame_content, frame_style = padder.pad(frame_content, frame_style)
    
    with torch.no_grad():
        frame_pred = filter_net(torch.cat([frame_content, frame_style], dim=1))
        ### local_net 
        if frame_id == first_processed_id:
            frame_o1 = frame_pred
            frame_o2 = frame_pred
            frame_p1 = frame_pred
        else:
            frame_p2 = frame_pred
            inputs = torch.cat((frame_p2, frame_o1, frame_p2, frame_p1), dim=1)
            output, _ = local_net(inputs, None)
            frame_o2 = frame_p2 + output
            frame_p1 = frame_p2
            frame_o1 = frame_o2

    frame_content, frame_style, frame_pred = tensor2img(frame_content), tensor2img(frame_style), tensor2img(frame_pred)
    frame_content = cv2.resize(frame_content, org_size, cv2.INTER_LINEAR)
    frame_style = cv2.resize(frame_style, org_size, cv2.INTER_LINEAR)
    frame_pred = cv2.resize(frame_pred, org_size, cv2.INTER_LINEAR)
    frame_concat = np.concatenate([frame_content, frame_style, frame_pred], axis=1)
    utils.save_img(frame_concat, "{}/{:05d}.png".format(output_folder, frame_id))
    utils.save_img(frame_pred, "{}/{:05d}.png".format(process_filter_dir, frame_id))

    frame_o2 = tensor2img(frame_o2)
    frame_o2 = cv2.resize(frame_o2, org_size, cv2.INTER_LINEAR)
    utils.save_img(frame_o2, "{}/{:05d}.png".format(output_final_dir, frame_id))


# save video
cmd =  "ffmpeg -y -r %s -i %s -crf 16 -r %s -pix_fmt yuv420p %s" % (str(opts.fps), os.path.join(output_folder, "%05d.png"), str(opts.fps), output_folder + ".mp4")
os.system(cmd) 
cmd =  "ffmpeg -y -r %s -i %s -crf 16 -r %s -pix_fmt yuv420p %s" % (str(opts.fps), os.path.join(process_filter_dir, "%05d.png"), str(opts.fps), process_filter_dir + ".mp4")
os.system(cmd) 
cmd =  "ffmpeg -y -r %s -i %s -crf 16 -r %s -pix_fmt yuv420p %s" % (str(opts.fps), os.path.join(output_final_dir, "%05d.png"), str(opts.fps), output_final_dir + ".mp4")
os.system(cmd)
