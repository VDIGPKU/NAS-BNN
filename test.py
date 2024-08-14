import argparse
import logging
import os
import os.path as osp
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import models
from utils import get_logger, tuple2cand

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('supernet', type=str)
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
parser.add_argument('--max-train-iters', type=int, default=10)
parser.add_argument('--train-batch-size', type=int, default=2048)
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
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
                    help='seed for initializing training.')
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


class DataIterator:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


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
        logger = get_logger(name='Test',
                            log_file=osp.join(args.logdir, 'test.log'),
                            log_level=logging.INFO)
        logger.info(args)
    else:
        logger = None
    if is_first_gpu(args, ngpus_per_node):
        logger.info(f"=> creating model '{args.arch}'")

    searched = torch.load(args.checkpoint)
    sub_path = searched['pareto_global'][args.ops]
    model = models.__dict__[args.arch](sub_path=tuple2cand(sub_path))
    # model.sub_path = model.smallest_cand
    # model.sub_path = model.biggest_cand

    flops, bitops, total_flops = model.get_ops()
    if is_first_gpu(args, ngpus_per_node):
        logger.info(f"=> creating model '{args.arch}'")
        logger.info(f"=> search arch '{sub_path}'")
        logger.info('=> flops {}M, bitops {}M, total flops {}M'.format(
            flops, bitops, total_flops))
    if os.path.isfile(args.supernet):
        if is_first_gpu(args, ngpus_per_node):
            logger.info(f"=> loading checkpoint '{args.supernet}'")
        checkpoint = torch.load(args.supernet, map_location='cpu')
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        model.load_state_dict(state_dict)
        if is_first_gpu(args, ngpus_per_node):
            logger.info(f"=> loaded checkpoint '{args.supernet}'")
    else:
        if is_first_gpu(args, ngpus_per_node):
            logger.info(f"=> no checkpoint found at '{args.supernet}'")
            exit(0)

    if args.dataset == 'imagenet':
        dummy_input = torch.randn((1, 3, 224, 224))
    else:
        dummy_input = torch.randn((1, 3, 32, 32))
    model.eval()
    model.to_static(dummy_input)

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
            args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
            args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    train_provider = DataIterator(train_loader)
    max_train_iters = args.max_train_iters

    print('calibrate running stats for BN....')
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.momentum = None  # cumulative moving average
            m.reset_running_stats()

    with torch.no_grad():
        for step in tqdm.tqdm(range(max_train_iters)):
            # print('train step: {} total: {}'.format(step,max_train_iters))
            images, target = train_provider.next()
            # print('get data',data.shape)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output, _ = model(images)
            del images, target, output
    device = next(model.parameters()).device
    top1 = torch.tensor([0.], device=device)
    top5 = torch.tensor([0.], device=device)
    total = torch.tensor([0.], device=device)

    print('starting test....')
    model.eval()

    for images, target in tqdm.tqdm(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        batchsize = images.shape[0]
        # print('get data',data.shape)

        output, _ = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1 += acc1.item() * batchsize
        top5 += acc5.item() * batchsize
        total += batchsize

        del images, target, output, acc1, acc5
    if args.distributed:
        dist.barrier()
        dist.all_reduce(top1)
        dist.all_reduce(top5)
        dist.all_reduce(total)
    top1, top5 = top1 / total, top5 / total

    if is_first_gpu(args, ngpus_per_node):
        logger.info('top1: {:.2f} top5: {:.2f}'.format(top1.item(),
                                                       top5.item()))


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
