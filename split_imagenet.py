import argparse
import os
import os.path as osp
import random
import shutil

import tqdm

random.seed(0)


def move_file(src_dir, dst_dir):
    pathDir = os.listdir(src_dir)

    picknumber = 50
    sample = random.sample(pathDir, picknumber)
    for name in sample:
        shutil.move(osp.join(src_dir, name), osp.join(dst_dir, name))


def link_file(src_dir, dst_dir):
    pathDir = os.listdir(src_dir)
    for name in pathDir:
        os.symlink(osp.join(src_dir, name), osp.join(dst_dir, name))


parser = argparse.ArgumentParser()
parser.add_argument('src_dir', metavar='DIR')
parser.add_argument('dst_dir', metavar='DIR')
args = parser.parse_args()

if __name__ == '__main__':
    train_dir = osp.join(args.src_dir, 'train')
    dst_train_dir = osp.join(args.dst_dir, 'train')
    dst_val_dir = osp.join(args.dst_dir, 'val')

    if not osp.exists(dst_train_dir):
        os.makedirs(dst_train_dir)
    else:
        print('err1')
        exit(1)

    if not osp.exists(dst_val_dir):
        os.makedirs(dst_val_dir)
    else:
        print('err2')
        exit(1)

    classes = os.listdir(train_dir)
    for c in tqdm.tqdm(classes):
        src_path = osp.join(train_dir, c)
        dst_train_path = osp.join(dst_train_dir, c)
        dst_val_path = osp.join(dst_val_dir, c)
        if not osp.exists(dst_train_path):
            os.makedirs(dst_train_path)
        else:
            print('err3')
            exit(1)
        if not osp.exists(dst_val_path):
            os.makedirs(dst_val_path)
        else:
            print('err4')
            exit(1)
        link_file(src_path, dst_train_path)
        move_file(dst_train_path, dst_val_path)
