import os
import numpy as np
# from utils import rrmse
import h5py
import matplotlib.pyplot as plt

import os
import h5py
import hdf5storage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import DatasetFromHdf5
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR

def rrmse(img_res, img_gt, mask_gt):
    """Calculate the relative RMSE"""
    error = img_res - img_gt
    error_relative = error / mask_gt
    rrmse = np.mean((np.sqrt(np.power(error_relative, 2))))
    return rrmse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_path = './test/input/'
gt_path = './test/gt/'
output_path = './test/output/'

model_path = './test/models/val-7-8-14_1.67_net_21epoch.pth'
input_chan = 20
input_img = 'bio'
input_img = 'blood'
numbers = [1, 2, 3, 4, 5]
# input_img = 'chart'
# numbers = [14, 7, 8, 11, 24]
dots = [(65 + 80 * ((i - 1) // 5), 65 + 80 * ((i - 1) % 5)) for i in numbers]
chan_label = 'chann+{}'.format(input_chan)
model_label = model_path.split('/')[3][:-4]

for img_name in sorted(os.listdir(input_path)):
    if input_img in img_name and chan_label in img_name:
        data = h5py.File(os.path.join(input_path, img_name))
        filtered = np.transpose(data.get('F_color_chart')).copy()
        f_w_length = np.squeeze(data.get('filtered_w_length'))

for img_name in sorted(os.listdir(output_path)):
    if input_img in img_name and chan_label in img_name:
        data = h5py.File(os.path.join(output_path, img_name))
        recon = np.squeeze(data.get('reconstruct')).copy()

for img_name in sorted(os.listdir(gt_path)):
    if input_img in img_name:
        data = h5py.File(os.path.join(gt_path, img_name))
        gt = np.transpose(data.get('N_color_chart')).copy()
        w_length = np.squeeze(data.get('w_length'))

print(filtered.shape, recon.shape, gt.shape)

mask_gt = gt.copy()
np.putmask(mask_gt, mask_gt == 0, 1000)

f, axes = plt.subplots(2, 3)
f.set_size_inches((17, 9))
f.suptitle('Spectrum Graph ({})'.format(model_label), fontsize=15)
plt.tight_layout(pad=2, h_pad=2.5)

error = abs((recon - gt)/mask_gt).mean(axis=2)
im = axes[0, 0].imshow(error)
plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[0, 0].set_title('relative error mean')

for i, dot in enumerate(dots):
    axes[0, 0].scatter(dot[0], dot[1], c='red', s=8, marker='*')
    axes[0, 0].text(dot[0] - 25, dot[1] - 10, dot, size=9, color='white')
    i = i + 1
    axes[i // 3, i % 3].plot(w_length, gt[dot[1], dot[0], :])
    axes[i // 3, i % 3].plot(w_length, recon[dot[1], dot[0], :])
    axes[i // 3, i % 3].plot(f_w_length, filtered[dot[1], dot[0], :])
    axes[i // 3, i % 3].set_title('{} - MRAE: {:.4f}'.format(dot, error[dot[1]][dot[0]]))
    axes[i // 3, i % 3].legend(['GT', 'Reconstruction', 'Filtered'])
    # axes[i//3, i%3].set_ylim([0, 1])

print(rrmse(recon, gt, mask_gt))

plt.show()


