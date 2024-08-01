import sys
import copy
from models.RFANet import BaseNet

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import utils
import matplotlib.pyplot as plt

import os, time
import numpy as np
from argparse import ArgumentParser
from torch.utils import tensorboard
from thop import profile
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output, output2, output3, output4 = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var) + BCEDiceLoss(output2, target_var) + BCEDiceLoss(output3, target_var) + \
               BCEDiceLoss(output4, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        # salEvalVal.addBatch(pred, target_var)
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

        if np.mod(iter, 200) == 1:
            vis_input = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred = utils.make_numpy_grid(pred[0:8])
            vis_gt = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'val_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, writer, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    total_batches = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output, output2, output3, output4 = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var) + BCEDiceLoss(output2, target_var) + BCEDiceLoss(output3, target_var) + \
               BCEDiceLoss(output4, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        # Computing F-measure and IoU on GPU
        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time),
                  end='')

        writer.add_scalar(f'train/f1', f1, iter + cur_iter)
        writer.add_scalar(f'train/lr', lr, iter + cur_iter)
        writer.add_scalar(f'train/ce', loss.data.item(), iter + cur_iter)

        if np.mod(iter, 200) == 1:
            vis_input = utils.make_numpy_grid(utils.de_norm(pre_img_var[0:8]))
            vis_input2 = utils.make_numpy_grid(utils.de_norm(post_img_var[0:8]))
            vis_pred = utils.make_numpy_grid(pred[0:8])
            vis_gt = utils.make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'train_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def trainValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet(3, 1)

    args.savedir = args.savedir + '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'
    args.vis_dir = args.savedir + '/Vis/'

    if args.file_root == 'LEVIR':
        args.file_root = "/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/LEVIR_256"
    elif args.file_root == 'WHU':
        args.file_root = '/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/WHU_256'
    elif args.file_root == 'DSIFN':
        args.file_root = '/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/DSIFN_256'
    elif args.file_root == 'BCDD':
        args.file_root = '/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/BCDD_256'
    elif args.file_root == 'SYSU':
        args.file_root = "/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/SYSU_256"
    elif args.file_root == 'CDD':
        args.file_root = "/media/cvpr/9ef49ce6-0406-42e1-85f2-9e767157906a/yzh/CD/CDD_256"
    elif args.file_root == 'quick_start':
        args.file_root = './samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    writer = tensorboard.SummaryWriter(args.savedir)

    if args.onGPU:
        model = model.cuda()

    input_data_A = torch.randn(1, 3, 256, 256).cuda()
    input_data_B = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, (input_data_A, input_data_B))
    print('Total network parameters (excluding idr): ' + str(params))
    print('Total network flops (excluding idr): ' + str(flops))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        # myTransforms.GaussianNoise(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main)

    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    val_data = myDataLoader.Dataset("val", file_root=args.file_root, transform=valDataset)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    # whether use multi-scale training

    max_batches = len(trainLoader)

    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    if args.resume is not None:
        args.resume = args.savedir + '/checkpoint.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * len(trainLoader)
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        logger = open(logFileLoc, 'w')
        logger.write('================ (%s) ================\n' % formatted_time)
        logger.write("\nParameters: %s \t\t" % params)
        logger.write("Flops: %s\n" % flops)
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                'Epoch', 'Kappa (val)', 'OA(val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    for epoch in range(start_epoch, args.max_epochs):

        lossTr, score_tr, lr = \
            train(args, trainLoader, model, optimizer, epoch, max_batches, writer, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        # evaluate on validation set
        if epoch == 0:
            continue
        if args.test:
            test_model = copy.deepcopy(model)
        lossVal, score_val = val(args, valLoader, model, epoch)
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                     (epoch, score_val['Kappa'], score_val['OA'], score_val['IoU'], score_val['F1'],
                      score_val['recall'], score_val['precision']))
        logger.flush()

        writer.add_scalar('val/Kappa', score_val['Kappa'], epoch)
        writer.add_scalar('val/IoU', score_val['IoU'], epoch)
        writer.add_scalar('val/F1', score_val['F1'], epoch)
        writer.add_scalar('val/recall', score_val['recall'], epoch)
        writer.add_scalar('val/precision', score_val['precision'], epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also
        model_file_name = args.savedir + 'best_model.pth'
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)

        writer.add_scalar('per-epoch/train-Loss', lossTr, epoch)
        writer.add_scalar('per-epoch/val-Loss', lossVal, epoch)
        writer.add_scalar('per-epoch/train-f1', score_tr['F1'], epoch)
        writer.add_scalar('per-epoch/val-f1', score_val['F1'], epoch)

        print("\tEpoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f" \
              % (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1']))

        # test per epoch
        if args.test:
            loss_test, score_test = val(args, testLoader, test_model, 0)
            writer.add_scalar('test/Kappa', score_test['Kappa'], epoch)
            writer.add_scalar('test/IoU', score_test['IoU'], epoch)
            writer.add_scalar('test/F1', score_test['F1'], epoch)
            writer.add_scalar('test/recall', score_test['recall'], epoch)
            writer.add_scalar('test/precision', score_test['precision'], epoch)

        torch.cuda.empty_cache()
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                 ('Test', score_test['Kappa'], score_test['OA'], score_test['IoU'], score_test['F1'],
                  score_test['recall'], score_test['precision']))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.write('\n================ (%s) ================' % formatted_time)
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | WHU | CDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=True, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')
    parser.add_argument('--test', type=bool, default=False, help='test on per epoch')

    args = parser.parse_args()
    print('Called with args:')
    print(args)
    trainValidateSegmentation(args)
