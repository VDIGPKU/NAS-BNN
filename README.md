# NAS-BNN [[arXiv]](https://arxiv.org/abs/2408.15484)

This repo contains the official implementation of **["NAS-BNN: Neural Architecture Search for Binary Neural Networks"](https://arxiv.org/abs/2408.15484)**.

## Open source model and searched result

- [Model](https://github.com/VDIGPKU/NAS-BNN/releases/download/v0.0.1/checkpoint.pth.tar)
- [Searched Result](https://github.com/VDIGPKU/NAS-BNN/releases/download/v0.0.1/info.pth.tar)

## Quick start

### Installation

```bash
conda create -n nasbnn python=3.9
pip install -r requirements.txt
```

### Data preparation

1. Preparation ImageNet dataset.

```
├── data
│   ├── ImageNet
│   │   ├── train
│   │   ├── val
│   │   ├── train_list.txt
│   │   ├── val_list.txt

```

2. Split ImageNet-1K train dataset into train/val datasets.

```bash
python split_imagenet.py ./data/ImageNet ./data/ImageNet_split
```

### Launch

Step 1: Training supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --dist-url 'tcp://127.0.0.1:29701' \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a superbnn -b 512 --lr 2.5e-3 --wd 5e-6 --epochs 512 \
    data/ImageNet_split ./work_dirs/nasbnn_exp
```

Step 2: Searching

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python search.py --dist-url 'tcp://127.0.0.1:29702' \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    --max-epochs 20 --population-num 512 --m-prob 0.2 --crossover-num 128 --mutation-num 128 \
    --ops-min 20 --ops-max 180 --step 2 --max-train-iters 10 --train-batch-size 2048 --test-batch-size 2048 \
    --dataset imagenet -a superbnn ./work_dirs/nasbnn_exp/checkpoint.pth.tar \
    data/ImageNet_split ./work_dirs/nasbnn_exp/search
```

Step 3: Testing

```bash
# Use --ops to specify the computational size of the model you want to test.
# The reasonable range is 20, 22, 24, ..., 180. E.g., --ops 100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --dist-url 'tcp://127.0.0.1:29701' \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset imagenet -a superbnn --ops [OPS] \
    --max-train-iters 10 --train-batch-size 2048 --test-batch-size 128 \
    ./work_dirs/nasbnn_exp/checkpoint.pth.tar data/ImageNet \
    ./work_dirs/nasbnn_exp/search/info.pth.tar ./work_dirs/nasbnn_exp/search/test
```

Step 4: Fine-tuning (optional)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_single.py --dist-url 'tcp://127.0.0.1:29701' \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    -a superbnn -b 512 --lr 1e-5 --wd 0 --epochs 25 --ops [OPS] \
    --pretrained ./work_dirs/nasbnn_exp/checkpoint.pth.tar \
    data/ImageNet ./work_dirs/nasbnn_exp/search/info.pth.tar ./work_dirs/nasbnn_exp/finetuned_opsxx
```

## License

The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.
