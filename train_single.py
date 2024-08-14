import argparse
import logging
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

import models as models
from utils import Cutout, get_logger, tuple2cand

# import wandb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('checkpoint',
                    type=str,
                    metavar='PATH',
                    help='path to searched checkpoint')
parser.add_argument('logdir', metavar='DIR')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='superbnn_cifar10',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: superbnn_cifar10)')
parser.add_argument('--dataset',
                    type=str,
                    default='imagenet',
                    help='imagenet | cifar10')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--ops', type=int, default=80)
parser.add_argument('--epochs',
                    default=25,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warm_up', action='store_true')
parser.add_argument('--warm_up-multiplier', default=1, type=float)
parser.add_argument('--warm_up-epochs', default=5, type=int)
parser.add_argument('--cutout',
                    action='store_true',
                    default=False,
                    help='use cutout')
parser.add_argument('--cutout-length',
                    type=int,
                    default=16,
                    help='cutout length')
parser.add_argument('-b',
                    '--batch-size',
                    default=512,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
lr_scheduler_choice = ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']
parser.add_argument('--lr-scheduler',
                    default='CosineAnnealingLR',
                    choices=lr_scheduler_choice)
parser.add_argument('--step-size',
                    default=30,
                    type=int,
                    help='step size of StepLR')
parser.add_argument('--gamma',
                    default=0.1,
                    type=float,
                    help='lr decay of StepLR or MultiStepLR')
parser.add_argument('--milestones',
                    default=[80, 120],
                    nargs='+',
                    type=int,
                    help='milestones of MultiStepLR')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--save-freq',
                    default=1,
                    type=int,
                    metavar='N',
                    help='save frequency (default: 1)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')


def is_first_gpu(args, ngpus_per_node):
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parser.parse_args()
    seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    if is_first_gpu(args, ngpus_per_node):
        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)
        logger = get_logger(name='Train',
                            log_file=osp.join(args.logdir, 'train.log'),
                            log_level=logging.INFO)
        logger.info(args)
        writer = SummaryWriter(osp.join(args.logdir, 'tf_logs'))
        # wandb.init(project="mbv2_cifar", entity="lzh", name=args.logdir)
        # wandb.config.update(args)
    else:
        logger = None
        writer = None
    searched = torch.load(args.checkpoint)
    sub_path = searched['pareto_global'][args.ops]

    # create model
    model = models.__dict__[args.arch](sub_path=tuple2cand(sub_path))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            if is_first_gpu(args, ngpus_per_node):
                logger.info("=> loading checkpoint '{}'".format(
                    args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.replace('module.', '')] = v
            model.load_state_dict(state_dict)
            if is_first_gpu(args, ngpus_per_node):
                logger.info("=> loaded checkpoint '{}'".format(
                    args.pretrained))
        else:
            if is_first_gpu(args, ngpus_per_node):
                logger.info("=> no checkpoint found at '{}'".format(
                    args.pretrained))

    if args.dataset == 'imagenet':
        dummy_input = torch.randn((1, 3, 224, 224))
    else:
        dummy_input = torch.randn((1, 3, 32, 32))
    model.eval()
    model.to_static(dummy_input)

    flops, bitops, total_flops = model.get_ops()
    if is_first_gpu(args, ngpus_per_node):
        logger.info(f"=> creating model '{args.arch}'")
        logger.info(f"=> search arch '{model.sub_path}'")
        logger.info('=> flops {}M, bitops {}M, total flops {}M'.format(
            flops, bitops, total_flops))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available
        # GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, val_transform)
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize
        ])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])
        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, val_transform)
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if (p.ndimension() == 4
                and 'conv' in pname) or (p.ndimension() == 2 and
                                         ('linear' in pname or 'fc' in pname)):
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(
        filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam([{
        'params': other_parameters
    }, {
        'params': weight_parameters,
        'weight_decay': args.weight_decay
    }],
                                 lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler_lr = get_lr_scheduler(optimizer, args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if is_first_gpu(args, ngpus_per_node):
                logger.info(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_lr.load_state_dict(checkpoint['scheduler'])
            if is_first_gpu(args, ngpus_per_node):
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if is_first_gpu(args, ngpus_per_node):
                logger.info("=> no checkpoint found at '{}'".format(
                    args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, logger, writer, ngpus_per_node,
                 args)
        return

    if args.warm_up:
        args.milestones = [i - args.warmup_epochs for i in args.milestones]

    # this zero gradient update is needed to avoid a warning message.
    optimizer.zero_grad()
    optimizer.step()

    # if is_first_gpu(args, ngpus_per_node):
    #     wandb.watch(model)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, logger, writer,
              ngpus_per_node, args)
        scheduler_lr.step()

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, logger, writer,
                              ngpus_per_node, args)

        # save checkpoint
        if is_first_gpu(args, ngpus_per_node):
            if writer is not None:
                writer.add_scalar('val/acc1', acc1, epoch)
                writer.add_scalar('val/acc5', acc5, epoch)
            # wandb.log({'val/acc1': acc1, 'val/acc5': acc5})
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler_lr.state_dict()
                }, args)
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler_lr.state_dict()
                    },
                    args,
                    name=f'epoch_{epoch + 1}.pth.tar')
    if is_first_gpu(args, ngpus_per_node):
        if writer is not None:
            writer.close()
        # wandb.save(osp.join(args.logdir, 'checkpoint.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, logger, writer,
          ngpus_per_node, args):
    if is_first_gpu(args, ngpus_per_node):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        column = [batch_time, data_time, losses, top1, top5]
        progress = ProgressMeter(logger,
                                 len(train_loader),
                                 column,
                                 prefix=f'Epoch: [{epoch}]')

    # switch to train mode
    model.train()
    if hasattr(model, 'module'):
        m = model.module
    else:
        m = model
    m.set_bin_activation()
    m.set_bin_weight()

    base_step = epoch * len(train_loader)
    optimizer.zero_grad()
    if is_first_gpu(args, ngpus_per_node):
        end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if is_first_gpu(args, ngpus_per_node):
            # measure data loading time
            data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        batchsize = torch.tensor(target.size(0)).cuda(args.gpu)
        if args.distributed:
            dist.barrier()
            dist.all_reduce(batchsize)

        optimizer.zero_grad()

        output, _ = model(images)
        loss = criterion(output, target)
        # do step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 *= target.size(0)
        acc5 *= target.size(0)
        if args.distributed:
            dist.barrier()
            dist.all_reduce(acc1)
            dist.all_reduce(acc5)
        acc1 /= batchsize
        acc5 /= batchsize
        if is_first_gpu(args, ngpus_per_node):
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], batchsize)
            top5.update(acc5[0], batchsize)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0 and i > 0:
                progress.display(i)
                if writer is not None:
                    writer.add_scalar('train/lr',
                                      optimizer.param_groups[0]['lr'],
                                      base_step + i)
                    writer.add_scalar('train/acc1', top1.avg, base_step + i)
                    writer.add_scalar('train/acc5', top5.avg, base_step + i)
                    writer.add_scalar('train/loss', loss.item(), base_step + i)
                # info_dict = {'train/lr': optimizer.param_groups[0]['lr'],
                #              'train/acc1': top1.avg,
                #              'train/acc5': top5.avg,
                #              'train/loss': loss_sum.item(),
                #              'train/loss_cls': loss_cls.item()}
                # wandb.log(info_dict)


def validate(val_loader, model, criterion, logger, writer, ngpus_per_node,
             args):
    if is_first_gpu(args, ngpus_per_node):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(logger,
                                 len(val_loader),
                                 [batch_time, losses, top1, top5],
                                 prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            batchsize = torch.tensor(target.size(0)).cuda(args.gpu)
            if args.distributed:
                dist.barrier()
                dist.all_reduce(batchsize)

            # compute output
            output, _ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 *= target.size(0)
            acc5 *= target.size(0)
            if args.distributed:
                dist.barrier()
                dist.all_reduce(acc1)
                dist.all_reduce(acc5)
            acc1 /= batchsize
            acc5 /= batchsize
            if is_first_gpu(args, ngpus_per_node):
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], batchsize)
                top5.update(acc5[0], batchsize)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and i > 0:
                    progress.display(i)
        if is_first_gpu(args, ngpus_per_node):
            # TODO: this should also be done with the ProgressMeter
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

    if is_first_gpu(args, ngpus_per_node):
        return top1.avg, top5.avg
    else:
        return -1, -1


def save_checkpoint(state, args, name='checkpoint.pth.tar'):
    filename = osp.join(args.logdir, name)
    torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, logger, num_batches, meters, prefix=''):
        self.logger = logger
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == 'CosineAnnealingLR':
        print('Use cosine scheduler')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'StepLR':
        print('Use step scheduler, step size: {}, gamma: {}'.format(
            args.step_size, args.gamma))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(
            args.milestones, args.gamma))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError
    if args.warm_up:
        print('Use warm_up scheduler')
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            total_epoch=args.warmup_epochs,
            after_scheduler=lr_scheduler)
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
    return lr_scheduler


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
