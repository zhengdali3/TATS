import numpy as np
import torch
import argparse

from tats import VideoData
from tats.utils import shift_dim
from tats.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser.add_argument('--npfile', type=str, default='')
args = parser.parse_args()

all_data_np = np.load(args.npfile)
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