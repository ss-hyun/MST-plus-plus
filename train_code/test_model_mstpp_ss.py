from __future__ import division

import os
import h5py
import hdf5storage

import numpy as np

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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_path = './test/models/val-7-8-14_1.67_net_21epoch.pth'
input_chan = 20
output_chan = 100
layer = 14
model_label = model_path.split('/')[3][:-4]
chan_label = 'chann+{}'.format(input_chan)

input_path = './test/input/'
output_path = './test/output/'
var_name = 'reconstruct'

save_point = torch.load(model_path)
model_param = save_point['state_dict']
method = 'mst_plus_plus'
model = model_generator(method).cuda()
model.load_state_dict(model_param)

for img_name in sorted(os.listdir(input_path)):
    if chan_label + '.mat' in img_name:
        print(img_name)
        img_path_name = os.path.join(input_path, img_name)
        data = h5py.File(img_path_name)
        input_img = data.get('F_color_chart')
        input_img = np.expand_dims(np.transpose(input_img, [0, 1, 2]), axis=0).copy()
        input_tensor = torch.Tensor(input_img).cuda()
        print(input_img.shape)

        input_tensor = [input_tensor]
        output_tensor = []

        if input_tensor[0].shape[2] == 450:
            input_tensor = [input_tensor[0][:, :, 0:105, :], input_tensor[0][:, :, 105:185, :],
                            input_tensor[0][:, :, 185:265, :], input_tensor[0][:, :, 265:345, :],
                            input_tensor[0][:, :, 345:450, :]]
        elif input_tensor[0].shape[2] == 530:
            input_tensor = [input_tensor[0][:, :, 0:105, :], input_tensor[0][:, :, 105:185, :],
                            input_tensor[0][:, :, 185:265, :], input_tensor[0][:, :, 265:345, :],
                            input_tensor[0][:, :, 345:425, :], input_tensor[0][:, :, 425:530, :]]

        for crop in input_tensor:
            img_res1 = model(crop)
            img_res2 = torch.flip(model(torch.flip(crop, (0, 2))), (0, 2))
            img_res3 = (img_res1 + img_res2) / 2
            output_tensor.append(img_res3.cpu().detach().numpy())

        output_tensor = np.concatenate(output_tensor, 2)
        print(output_tensor.shape)

        mat_name = 'recon_' + model_label + img_name.replace('input', '')
        mat_dir = os.path.join(output_path, mat_name)

        hdf5storage.savemat(mat_dir, {var_name: output_tensor}, format='7.3',
                            store_python_metadata=True)
