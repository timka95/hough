import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#import PyramidNet_modified as PYRM

import stage1_Loader
import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value

from lcnn.models.HT import hough_transform
from models.houghtransform_RAL_TIMKA import ht
from lcnn.models.multitask_learner import MultitaskHead , MultitaskLearner

from lcnn.config import C, M




import pickle
from datetime import datetime
import torch.nn as nn
import numpy as np

from torch import autograd

from torch.autograd import Variable

from Class_losses.losses_new import MultiLoss_new
from Class_losses.sampler_new import HichemSampler
from Class_losses.sampler_new import NghSampler2
from Class_losses.triplet_loss_new import  TripletLoss

from Class_losses.MSE_loss_new import MSELoss_new

import scipy.io as sio
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net_type', default='PyramidNet', type=str, help='networktype: resnet, resnext, densenet, pyamidnet, and so on')

parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#parser.add_argument('--resume', default='/home/hichem/MainProjects/R2D2_RAL_network_style/Pycharm_projects/training_128_detection_RAL/runs/HT_128_detect_mse_rel_b20/checkpoint_97.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


#parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.025, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')

parser.add_argument('--expname', default='Omni_Stage1_detection_512_wd1e_minus_6_basic', type=str, help='name of experiment')



# TIMKAAA
device_name = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device_name = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
else:
    print("CUDA is not available")
device = torch.device(device_name)


# TIMKAA



parser.set_defaults(bottleneck=False)
parser.set_defaults(verbose=True)



best_err1 = 100
best_err5 = 100
global step
step=0
global filetowrite

def main():
    global args, best_err1, best_err5
    global filetowrite
    global step
    ####################################Dimension of the input image--> hough #######################################################################
    global Img_dimension
    Img_dimension = 128
    Img_dimension1 = 128
    theta_res = 3
    ####################################Dimension of the input image--> hough #######################################################################

    args = parser.parse_args()
    #if args.tensorboard:
    configure("runs/%s"%(args.expname))
    namefile = "runs/%s/consol.txt"%(args.expname)
    filetowrite = open(namefile, 'w')


    #normalize = transforms.Normalize(mean=[0.54975, 0.60135, 0.63516], std=[0.34359,0.34472,0.34718])
    normalize = transforms.Normalize(mean=[0.492967568115862], std=[0.272086182765434])
    #mean_image =0.492967568115862
    #std_image = 0.272086182765434

    transform_train = transforms.Compose([ transforms.ToTensor(), normalize, ])

    train_loader = torch.utils.data.DataLoader(
            #stage1_Loader.LyftLoader('/home/hichem/OmniProject/data/equirectangular_512_training_berm_1/',img_count =0,  train=True, transform=transform_train,bsize= args.batch_size),
            #batch_size=1, shuffle=True, num_workers=4, pin_memory=False)

            stage1_Loader.LyftLoader('/home/hichem/OmniProject/data/equirectangular_512_training_berm_1/',img_count =0,  train=True, transform=transform_train,bsize= args.batch_size),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=False)

    print("=> creating model")

    if os.path.isfile(C.io.vote_index):
        print('load vote_index ... ')
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        print('compute vote_index ... ')
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('vote_index loaded', vote_index.shape)


    model = ht(
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        vote_index=vote_index,
        batches_size=args.batch_size,

        depth=M.depth,
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )

    model = model.to('cuda')
    model = torch.nn.DataParallel(model).cuda()


    #print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
            args.start_epoch = checkpoint['epoch']+1
            step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 0

    cudnn.benchmark = True

    default_loss = """MultiLoss_new(1, MSELoss_new(), )"""
    #default_loss = """MultiLoss_new(1, MSELoss_new())"""

    MultiLoss = eval(default_loss)

    print("\n>> Creating loss functions")

    #torch.autograd.set_detect_anomaly(True)

    #for param in model.parameters():
    #    print(param.dtype)

    for epoch in range(args.start_epoch, args.epochs+1):
        #if args.distributed:
            #train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        print()
        print("here 2 adjust_learning_rate(optimizer, epoch)")
        print()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,MultiLoss,args.batch_size)
        save_checkpoint2({
            'epoch': epoch,
            'step': step,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        },epoch)
        print("here 3)")

    filetowrite.close()
    #evaluate(val_loader, model)
    # print ('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    print ('end')


def train(train_loader, model, criterion, optimizer, epoch, MultiLoss, batches_size):
    global step
    global filetowrite
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mse_losses = AverageMeter()
    triplet_losses = AverageMeter()
    # switch to train mode
    model.train()
    #torch.set_grad_enabled(True)
    lossdata = 0
    mse_lossdata = 0
    triplet_lossdata = 0

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]

    for i, (input1, target_lines, image_name) in enumerate(train_loader):
        # measure data loading time

        #TIMKA smaller data
        if i >= 10:  # Change this value to the desired number of iterations
            break

        #TIMKA smaller data

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        input1 = input1.cuda()
        target_lines = target_lines.cuda()

        line_detected = model.forward(input1)

        #plt.imshow(line_detected[0, 0, :, :].cpu().detach().numpy(), cmap='gray')

        inputs = {
            "lines": line_detected,
            "lines_gt": target_lines,
            "batches_size": batches_size,
        }

        allvars = dict(inputs)
        loss, details = MultiLoss.forward_all(**allvars)

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        lossdata = lossdata + loss.data.detach().cpu()

        mse_lossdata = mse_lossdata + details['loss_difference']

        losses.update(loss.data.detach().cpu().numpy(), 1)

        mse_losses.update(details['loss_difference'], 1)

        # compute gradient and do SGD step


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
                  .format(
                   epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                    loss=losses))
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
                  .format(
                   epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                    loss=losses),file=filetowrite)
    # log to TensorBoard
    #if args.tensorboard:
        log_value('train_loss_all', loss.data.detach().cpu().numpy(), step)
        step = step + 1
    log_value('train_loss', losses.avg, epoch)
    log_value('train_mse', mse_losses.avg, epoch)




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs_TIMKA/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')

def save_checkpoint2(state, epoch_s,filename='checkpoint_'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename1 = directory  +  filename +str(epoch_s) + '.pth.tar'
    torch.save(state, filename1)
    filename2 = directory  +  filename + '.pth.tar'
    torch.save(state, filename2)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #args.dataset == ('hichem')
    #lr = args.lr * (0.1 ** (epoch // 30))
    #if args.tensorboard:
    lr = args.lr
    log_value('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
