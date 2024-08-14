import torch
import torch.nn.functional as F


def _sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def adaptive_add(src, residual):
    src_c, src_wh = src.shape[1], src.shape[2]
    residual_c, residual_wh = residual.shape[1], residual.shape[2]
    if src_wh != residual_wh:
        if src_wh == residual_wh // 2:
            residual = F.avg_pool2d(residual, 2, stride=2)
        else:
            raise NotImplementedError
    if src_c == residual_c:
        out = src + residual
    else:
        out = src + torch.cat(
            [residual for _ in range(round(src_c / residual_c + 0.5))],
            dim=1)[:, :src_c]
    return out


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """loss function measured in L_p Norm."""
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
