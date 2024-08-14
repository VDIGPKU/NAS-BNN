# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation adapted from Slimmable
# https://github.com/JiahuiYu/slimmable_networks
import torch


class KLLossSoft(torch.nn.modules.loss._Loss):
    r""" inplace distillation for image classification
            output: output logits of the student network
            target: output logits of the teacher network
            T: temperature
            KL(p||q) = Ep \log p - \Ep log q
    """

    def forward(self,
                output,
                soft_logits,
                target=None,
                temperature=1.,
                alpha=0.9):
        origin_output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = temperature * temperature * (
            -torch.sum(soft_target_prob * output_log_prob, dim=1))
        if target is not None:
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            origin_output_log_prob = origin_output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, origin_output_log_prob).squeeze()
            loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
        else:
            loss = kd_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
