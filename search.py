import argparse
import logging
import os
import os.path as osp
import random
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models as models
from utils import cand2tuple, get_logger, tuple2cand

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('supernet', type=str)
parser.add_argument('data', metavar='DIR', help='path to dataset')
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
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--population-num', type=int, default=512)
parser.add_argument('--m-prob', type=float, default=0.2)
parser.add_argument('--crossover-num', type=int, default=128)
parser.add_argument('--mutation-num', type=int, default=128)
parser.add_argument('--ops-min', type=float, default=40)
parser.add_argument('--ops-max', type=float, default=250)
parser.add_argument('--step', type=float, default=10)
parser.add_argument('--max-train-iters', type=int, default=10)
parser.add_argument('--train-batch-size', type=int, default=256)
parser.add_argument('--test-batch-size', type=int, default=256)
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


class EvolutionSearcher:

    def __init__(self, model, logger, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node

        self.max_epochs = args.max_epochs
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.ops_min = args.ops_min
        self.ops_max = args.ops_max
        self.step = args.step

        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)

        self.logger = logger

        self.model = model
        if hasattr(model, 'module'):
            self.m = model.module
        else:
            self.m = model

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
            val_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
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

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_name = os.path.join(args.logdir, 'info.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.epoch = 0
        self.candidates = []
        self.pareto_global = {}

    def save_checkpoint(self):
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['epoch'] = self.epoch
        info['pareto_global'] = self.pareto_global
        torch.save(info, self.checkpoint_name)
        if is_first_gpu(self.args, self.ngpus_per_node):
            self.logger.info('Save checkpoint to {}'.format(
                self.checkpoint_name))

    def is_legal(self, cand):
        cand_tuple = cand2tuple(cand)
        if cand_tuple not in self.vis_dict:
            self.vis_dict[cand_tuple] = {}
        info = self.vis_dict[cand_tuple]
        if 'visited' in info:
            return False
        if 'ops' not in info:
            _, _, info['ops'] = self.m.get_ops(cand)
        info['acc'], _ = get_cand_acc(self.model, cand, self.train_loader,
                                      self.val_loader, self.args)
        info['visited'] = True
        return True

    def get_random(self, num):
        cnt = 0
        while len(self.candidates) < num:
            if cnt == 0:
                cnt += 1
                cand = self.m.smallest_cand
            elif cnt == 1:
                cnt += 1
                cand = self.m.biggest_cand
            else:
                cand = self.m.get_random_range_cand(self.ops_min, self.ops_max)
            if self.args.distributed:
                dist.barrier()
                dist.broadcast(cand, 0)
            if not self.is_legal(cand):
                continue
            cand_tuple = cand2tuple(cand)
            self.candidates.append(cand_tuple)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('random {}/{}: {}'.format(
                    len(self.candidates), num, cand_tuple))

    def get_mutation(self):
        res = []
        while len(res) < self.mutation_num:
            ori_cand = tuple2cand(
                random.choice(list(self.pareto_global.values())))
            cand = ori_cand.clone()
            search_space = self.m.search_space
            stage_first = [0] * len(search_space)

            for stage_num in range(len(search_space)):
                num_blocks = search_space[stage_num][1]
                if stage_num > 0:
                    stage_first[stage_num] += stage_first[stage_num - 1] + max(
                        search_space[stage_num - 1][1])
                if random.random() < self.m_prob:
                    d = random.choice(num_blocks)
                    for block_num in range(max(num_blocks)):
                        if block_num < d:
                            cand[stage_first[stage_num] + block_num,
                                 0] = stage_num
                            cand[stage_first[stage_num] + block_num,
                                 1] = block_num
                        else:
                            cand[stage_first[stage_num] + block_num, 0] = -1
                            cand[stage_first[stage_num] + block_num, 1] = -1

            for i in range(cand.shape[0]):
                stage_num = cand[i][0]
                block_num = cand[i][1]
                if stage_num == -1 or block_num == -1:
                    continue
                if random.random() < self.m_prob:
                    if i == 0:
                        last_channel = -1
                    else:
                        last_channel = cand[i - 1][2]
                    channel_cand = torch.tensor(
                        search_space[stage_num][0][block_num][0])
                    channel_cand = channel_cand[
                        channel_cand >= last_channel].tolist()
                    cand[i][2] = random.choice(channel_cand)
                    cand[i][3] = random.choice(
                        search_space[stage_num][0][block_num][1])
                    cand[i][4] = random.choice(
                        search_space[stage_num][0][block_num][2])
            device = next(self.m.parameters()).device
            cand = cand.to(device)
            if self.args.distributed:
                dist.barrier()
                dist.broadcast(cand, 0)
            _, _, ops = self.m.get_ops(cand)
            if not (self.ops_min <= ops <= self.ops_max):
                continue
            if not self.is_legal(cand):
                continue
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('mutation {}/{}, {}'.format(
                    len(res), self.mutation_num, cand_tuple))
        return res

    def get_crossover(self):
        res = []
        while len(res) < self.crossover_num:
            cand1 = tuple2cand(random.choice(list(
                self.pareto_global.values())))
            cand2 = tuple2cand(random.choice(list(
                self.pareto_global.values())))
            search_space = self.m.search_space
            d_list = []
            for i in range(len(search_space)):
                d1 = (cand1[cand1[:, 0] == i]).shape[0]
                d2 = (cand2[cand2[:, 0] == i]).shape[0]
                d_list.append(random.choice([d1, d2]))
            mask = torch.rand_like(cand1.float()).round().int()
            cand = mask * cand1 + (1 - mask) * cand2
            if any(cand[:, 2].sort().values != cand[:, 2]):
                continue

            stage_first = [0] * len(search_space)
            for stage_num in range(len(search_space)):
                num_blocks = search_space[stage_num][1]
                if stage_num > 0:
                    stage_first[stage_num] += stage_first[stage_num - 1] + max(
                        search_space[stage_num - 1][1])
                for block_num in range(max(num_blocks)):
                    if block_num < d_list[stage_num]:
                        cand[stage_first[stage_num] + block_num, 0] = stage_num
                        cand[stage_first[stage_num] + block_num, 1] = block_num
                    else:
                        cand[stage_first[stage_num] + block_num, 0] = -1
                        cand[stage_first[stage_num] + block_num, 1] = -1
            device = next(self.m.parameters()).device
            cand = cand.to(device)
            if self.args.distributed:
                dist.barrier()
                dist.broadcast(cand, 0)
            _, _, ops = self.m.get_ops(cand)
            if not (self.ops_min <= ops <= self.ops_max):
                continue
            if not self.is_legal(cand):
                continue
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('crossover {}/{}, {}'.format(
                    len(res), self.crossover_num, cand_tuple))
        return res

    def update_frontier(self):
        for cand_tuple in self.candidates:
            acc, ops = self.vis_dict[cand_tuple]['acc'], self.vis_dict[
                cand_tuple]['ops']
            f = int(round(ops / self.args.step) * self.args.step)
            if f not in self.pareto_global or self.vis_dict[
                    self.pareto_global[f]]['acc'] < acc:
                self.pareto_global[f] = cand_tuple

    def search(self):
        if is_first_gpu(self.args, self.ngpus_per_node):
            self.logger.info('population_num = {} mutation_num = {} '
                             'crossover_num = {} max_epochs = {}'.format(
                                 self.population_num, self.mutation_num,
                                 self.crossover_num, self.max_epochs))
            self.logger.info('Init')
        self.get_random(self.population_num)
        self.update_frontier()

        while self.epoch < self.max_epochs:
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('Epoch: {}/{}'.format(self.epoch,
                                                       self.max_epochs))

            self.memory.append([])
            for cand_tuple in self.candidates:
                self.memory[-1].append(cand_tuple)

            mutation = self.get_mutation()
            crossover = self.get_crossover()

            self.candidates = mutation + crossover
            self.update_frontier()

            if is_first_gpu(self.args, self.ngpus_per_node):
                ops_stages = sorted(list(self.pareto_global.keys()))
                for s in ops_stages:
                    cand_tuple = self.pareto_global[s]
                    self.logger.info(
                        'OPs Stage.{}, {}, Top-1 acc: {:.2f}, OPs: {:.2f}'.
                        format(s, cand_tuple, self.vis_dict[cand_tuple]['acc'],
                               self.vis_dict[cand_tuple]['ops']))
            self.epoch += 1

        self.save_checkpoint()


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
        logger = get_logger(name='Search',
                            log_file=osp.join(args.logdir, 'search.log'),
                            log_level=logging.INFO)
        logger.info(args)
    else:
        logger = None
    if is_first_gpu(args, ngpus_per_node):
        t = time.time()
        logger.info(f"=> creating model '{args.arch}'")
    model = models.__dict__[args.arch]()

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

    searcher = EvolutionSearcher(model, logger, args, ngpus_per_node)
    searcher.search()
    if is_first_gpu(args, ngpus_per_node):
        logger.info('total searching time = {:.2f} hours'.format(
            (time.time() - t) / 3600))


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


def no_grad_wrapper(func):

    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


@no_grad_wrapper
def get_cand_acc(model, cand, train_loader, val_loader, args):

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

    train_provider = DataIterator(train_loader)
    max_train_iters = args.max_train_iters

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.momentum = None  # cumulative moving average
            m.reset_running_stats()

    with torch.no_grad():
        for step in range(max_train_iters):
            images, _ = train_provider.next()
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            model(images, cand)

    device = next(model.parameters()).device
    top1 = torch.tensor([0.], device=device)
    top5 = torch.tensor([0.], device=device)
    total = torch.tensor([0.], device=device)

    model.eval()
    for images, target in val_loader:
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        batchsize = images.shape[0]
        # print('get data',data.shape)

        output, _ = model(images, cand)

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

    return top1.item(), top5.item()


if __name__ == '__main__':
    main()
