import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataset import DatasetFromHdf5
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime

'''
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
'''

# load dataset
print("\nloading dataset ...")
train_data = DatasetFromHdf5('./data/train_val-7-8-14_input+chann+20.h5')
print(f"Iteration per epoch: {len(train_data)}")
val_data = DatasetFromHdf5('./data/valid_val-7-8-14_input+chann+20.h5')
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = len(train_data)
init_lr = 4e-4
end_epoch = 300
total_iteration = per_epoch_iteration * end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
method = 'mst_plus_plus'
model = model_generator(method).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
out_dir = './MST++_' + date_time
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(out_dir, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = ''
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000
    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=64, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_data, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration + 1
            # if iteration % (per_epoch_iteration/10) == 0:
            #     print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
            #           % (iteration, total_iteration, lr, losses.avg))
            if iteration % per_epoch_iteration == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                # Save model
                if torch.abs(
                        mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % (per_epoch_iteration*10) == 0:
                    print(f'Saving to {out_dir}')
                    save_checkpoint(out_dir, (iteration // per_epoch_iteration), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                print(" Iter[%06d/%06d], Epoch[%04d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (
                          iteration, total_iteration, iteration // per_epoch_iteration, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%04d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                            "Test RMSE: %.9f, Test PSNR: %.9f " % (
                                iteration, iteration // per_epoch_iteration, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
    return 0


# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == '__main__':
    main()
    print(torch.__version__)
