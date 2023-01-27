import sys
sys.path.insert(0, '/userhome/42/msd21003/TATS')

import numpy as np
import torch
import argparse

from tats import VideoData
from tats.utils import shift_dim
from tats.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
args = parser.parse_args()

# class Args:
#     data_path = '/datasets01/Kinetics400_Frames/videos'
#     dataset = 'taichi'
#     sequence_length=16
#     resolution=64
#     batch_size=32
#     num_workers=8
#     image_channels=3
#     smap_cond=0
#     spatial_length=15
#     sample_every_n_frames=1
#     image_folder=False
    
# args=Args()

npFile = "/userhome/42/msd21003/result/numpy_files/" + args.dataset + "/topp0.92_topk2048_run0_eval.npy"
args.data_path = "/userhome/42/msd21003/" + args.dataset

all_data_np = np.load(npFile)
device = torch.device('cuda')
i3d = load_fvd_model(device)
data = VideoData(args)
loader = data.train_dataloader()
real_embeddings = []
print('computing fvd embeddings for real videos')
for batch in loader:
    real_embeddings.append(
        get_fvd_logits(shift_dim((batch['video'] + 0.5) * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
    if len(real_embeddings) * args.batch_size >= 2048: break
print('caoncat fvd embeddings for real videos')
real_embeddings = torch.cat(real_embeddings, 0)[:2048]
print('computing fvd embeddings for fake videos')
fake_embeddings = []
n_batch = all_data_np.shape[0] // args.batch_size
for i in range(n_batch):
    fake_embeddings.append(
        get_fvd_logits(all_data_np[i * args.batch_size:(i + 1) * args.batch_size], i3d=i3d, device=device))
print('caoncat fvd embeddings for fake videos')
fake_embeddings = torch.cat(fake_embeddings, 0)[:2048]
print('FVD = %.2f' % (frechet_distance(fake_embeddings, real_embeddings)))
print('KVD = %.2f' % (polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())))