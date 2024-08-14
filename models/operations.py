import torch
import torch.nn as nn


class SignFuncA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        indicate_small = (x < -1).float()
        indicate_big = (x > 1).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_x = (-2 * x.sign() * x + 2) * indicate_middle * grad_output
        return grad_x


class SignFuncW(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bound=1):
        ctx.save_for_backward(x)
        ctx.other = bound
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        bound = ctx.other
        indicate_small = (x < -bound).float()
        indicate_big = (x > bound).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_x = indicate_middle * grad_output
        return grad_x, None


class BinaryActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.is_bin = True

    def forward(self, x):
        if self.is_bin:
            out = SignFuncA.apply(x)
        else:
            out = x
        return out
