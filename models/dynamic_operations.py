import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import _sub_filter_start_end, grad_scale, lp_loss, round_pass
from .operations import SignFuncW


class DynamicLearnableBias(nn.Module):

    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(1, max_channels, 1, 1)))
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        if self.static:
            out = x + self.bias_s.expand_as(x)
        else:
            out = x + self.bias[:, :feature_dim].expand_as(x)
        return out

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        self.register_parameter('bias_s',
                                nn.Parameter(self.bias[:, :feature_dim].data))
        self.bias.requires_grad = False
        return self.forward(x)


class DynamicQConv2d(nn.Module):

    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups_list,
                 w_bit,
                 a_bit,
                 wh,
                 stride=1,
                 dilation=1,
                 channel_wise=True):
        super().__init__()

        self.max_inp = max_inp
        self.max_oup = max_oup

        self.ks_list = list(set(ks_list))
        self.ks_list.sort()  # e.g., [3, 5, 7]
        self.max_ks = max(ks_list)
        self.groups_list = groups_list
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.wh = wh
        self.stride = stride
        self.dilation = dilation
        shape = (self.max_oup, self.max_inp // min(groups_list), self.max_ks,
                 self.max_ks)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        if w_bit != 32:
            if channel_wise:
                self.scale_channel = max_oup
            else:
                self.scale_channel = 1
            self.register_parameter(
                'w_scale', nn.Parameter(torch.ones(self.scale_channel)))
            self.register_buffer('w_init_state', torch.zeros(1))
            self.w_min_q = -1 * 2**(w_bit - 1)
            self.w_max_q = 2**(w_bit - 1) - 1

        if a_bit != 32 and a_bit is not None:
            self.register_parameter('a_scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('a_zero_point',
                                    nn.Parameter(torch.ones(1)))
            self.register_buffer('a_init_state', torch.zeros(1))
            self.a_min_q = 0
            self.a_max_q = 2**a_bit - 1

        if len(self.ks_list) > 1:
            # register scaling parameters
            # matrix_7to5, matrix_5to3
            scale_params = {}
            for i in range(len(self.ks_list) - 1):
                ks_small = self.ks_list[i]
                ks_larger = self.ks_list[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params['matrix_%s' % param_name] = nn.Parameter(
                    torch.eye(ks_small**2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)
        self.eps = torch.finfo(torch.float32).eps
        self.ops_memory = {}
        self.reset_parameters()
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_active_filter(self, weight, inp, oup, ks, groups):
        start, end = _sub_filter_start_end(self.max_ks, ks)
        filters = weight[:oup, :inp, start:end, start:end]
        if ks < self.max_ks:
            start_filter = weight[:oup, :inp, :, :]  # start with max kernel
            for i in range(len(self.ks_list) - 1, 0, -1):
                src_ks = self.ks_list[i]
                if src_ks <= ks:
                    break
                target_ks = self.ks_list[i - 1]
                start, end = _sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0),
                                                   _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    getattr(self, 'matrix_%dto%d' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1),
                                                   target_ks**2)
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1), target_ks,
                                                   target_ks)
                start_filter = _input_filter
            filters = start_filter
        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_inp = inp // groups
        sub_ratio = filters.size(1) // sub_inp
        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_inp
            filter_crops.append(sub_filter[:, start:start + sub_inp, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, sub_path):
        assert x.shape[-1] == self.wh
        oup, ks, groups = sub_path
        inp = x.shape[1]
        if oup == -1:
            oup = inp
        if self.static:
            weight = self.weight_s
            w_scale = self.w_scale_s
        else:
            weight = self.get_active_filter(self.weight, inp, oup, ks,
                                            groups).contiguous()
            w_scale = self.w_scale[:oup]

        if self.a_bit != 32 and self.a_bit is not None:
            if self.training and self.a_init_state == 0:
                self.a_init_state.fill_(1)
                x_detach = x.detach()
                x_max = x_detach.max()
                x_min = x_detach.min()
                best_score = 1e+10
                for i in range(80):
                    new_ub = x_max * (1.0 - (i * 0.01))
                    new_lb = x_min * (1.0 - (i * 0.01))
                    new_scale = (new_ub - new_lb) / (2**self.a_bit - 1)
                    new_zero_point = new_lb
                    x_q = (torch.round(
                        (x_detach - new_zero_point) / new_scale)).clamp(
                            self.a_min_q,
                            self.a_max_q) * new_scale + new_zero_point
                    score = lp_loss(x_detach, x_q, p=2.0, reduction='all')
                    if score < best_score:
                        best_score = score
                        res_scale = new_scale
                        # res_zero_point = new_zero_point
                self.a_scale.data.copy_(res_scale)
                self.a_zero_point.data.copy_(new_zero_point)
            self.a_scale.data.clamp_(self.eps)
            g = 1.0 / math.sqrt(x.numel() * self.a_max_q)
            cur_a_scale = grad_scale(self.a_scale, g)
            cur_a_zero_point = grad_scale(self.a_zero_point, g)
            x = round_pass((x - cur_a_zero_point) / cur_a_scale).clamp(
                self.a_min_q, self.a_max_q) * cur_a_scale + cur_a_zero_point
        if self.w_bit != 32:
            if self.training and self.w_init_state == 0:
                self.w_init_state.fill_(1)
                w_detach = weight.detach()
                if self.scale_channel == 1:
                    w_mean = w_detach.mean()
                    w_std = w_detach.std()
                else:
                    dim = [i for i in range(1, w_detach.dim())]
                    w_mean = w_detach.mean(dim=dim)
                    w_std = w_detach.std(dim=dim)
                v1 = torch.abs(w_mean - 3 * w_std)
                v2 = torch.abs(w_mean + 3 * w_std)
                w_scale.data.copy_(torch.max(v1, v2) / 2**(self.w_bit - 1))
            w_scale.data.clamp_(self.eps)
            g = 1.0 / math.sqrt(weight.numel() * self.w_max_q)
            cur_w_scale = grad_scale(w_scale, g)
            if self.scale_channel != 1:
                cur_w_scale = cur_w_scale.view(oup, 1, 1, 1)
            weight = round_pass(weight / cur_w_scale).clamp(
                self.w_min_q, self.w_max_q) * cur_w_scale
        padding = (ks - 1) // 2 * self.dilation
        out = F.conv2d(x,
                       weight,
                       None,
                       stride=self.stride,
                       padding=padding,
                       dilation=self.dilation,
                       groups=groups)
        return out

    def to_static(self, x, sub_path):
        self.static = True
        assert x.shape[-1] == self.wh
        oup, ks, groups = sub_path
        inp = x.shape[1]
        if oup == -1:
            oup = inp
        weight = self.get_active_filter(self.weight, inp, oup, ks,
                                        groups).contiguous()
        self.weight.requires_grad = False
        if hasattr(self, 'matrix_7to5'):
            self.matrix_7to5.requires_grad = False
        if hasattr(self, 'matrix_5to3'):
            self.matrix_5to3.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(weight))
        if self.w_bit != 32:
            if self.scale_channel != 1:
                w_scale = self.w_scale[:oup]
            else:
                w_scale = self.w_scale
            self.w_scale.requires_grad = False
            self.register_parameter('w_scale_s', nn.Parameter(w_scale))
        return self.forward(x, sub_path)

    def get_flops_bitops(self, sub_path):
        assert self.max_inp == 3
        if tuple(sub_path) not in self.ops_memory:
            pre_channels = 3
            channels, ks, groups = sub_path
            bitops = 0.0
            if self.a_bit is None:
                factor = (64 / math.sqrt(self.w_bit * 8))
            else:
                factor = (64 / math.sqrt(self.w_bit * self.a_bit))
            flops = (ks * ks * pre_channels // groups * channels * self.wh //
                     self.stride * self.wh // self.stride // factor)
            self.ops_memory[tuple(sub_path)] = (flops / 1e6, bitops / 1e6)
        return self.ops_memory[tuple(sub_path)]


class DynamicFPLinear(nn.Module):

    def __init__(self, max_inp, oup):
        super().__init__()

        self.max_inp = max_inp
        self.oup = oup

        shape = (oup, max_inp)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_parameter('bias', nn.Parameter(torch.rand(oup)))
        self.reset_parameters()
        self.ops_memory = {}
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.static:
            filters = self.weight_s
        else:
            inp = x.shape[1]
            filters = self.weight[:, :inp].contiguous()
        out = F.linear(x, filters, self.bias)
        return out

    def to_static(self, x):
        self.static = True
        inp = x.shape[1]
        filters = self.weight[:, :inp].contiguous()
        self.weight.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(filters))
        return self.forward(x)

    def get_flops_bitops(self, pre_sub_path):
        if tuple(pre_sub_path) not in self.ops_memory:
            pre_channels = pre_sub_path[0]
            bitops = 0.0
            flops = self.oup * (pre_channels + 1)
            self.ops_memory[tuple(pre_sub_path)] = (flops / 1e6, bitops / 1e6)
        return self.ops_memory[tuple(pre_sub_path)]


class DynamicQLinear(nn.Module):

    def __init__(self, max_inp, max_oup, w_bit, a_bit, channel_wise=True):
        super().__init__()

        self.max_inp = max_inp
        self.max_oup = max_oup
        self.w_bit = w_bit
        self.a_bit = a_bit

        shape = (max_oup, max_inp)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_parameter('bias', nn.Parameter(torch.rand(max_oup)))

        if w_bit != 32:
            if channel_wise:
                self.scale_channel = max_oup
            else:
                self.scale_channel = 1
            self.register_parameter(
                'w_scale', nn.Parameter(torch.ones(self.scale_channel)))
            self.register_buffer('w_init_state', torch.zeros(1))
            self.w_min_q = -1 * 2**(w_bit - 1)
            self.w_max_q = 2**(w_bit - 1) - 1

        if a_bit != 32:
            self.register_parameter('a_scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('a_zero_point',
                                    nn.Parameter(torch.ones(1)))
            self.register_buffer('a_init_state', torch.zeros(1))
            self.a_min_q = 0
            self.a_max_q = 2**a_bit - 1

        self.eps = torch.finfo(torch.float32).eps

        self.reset_parameters()
        self.ops_memory = {}
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, oup=None):
        if self.static:
            weight = self.weight_s
            bias = self.bias_s
            w_scale = self.w_scale_s
            oup = self.weight_s.shape[0]
        else:
            inp = x.shape[1]
            if oup is None:
                oup = self.weight.shape[0]
            weight = self.weight[:oup, :inp].contiguous()
            bias = self.bias[:oup]
            w_scale = self.w_scale[:oup]

        if self.a_bit != 32:
            if self.training and self.a_init_state == 0:
                self.a_init_state.fill_(1)
                x_detach = x.detach()
                x_max = x_detach.max()
                x_min = x_detach.min()
                best_score = 1e+10
                for i in range(80):
                    new_ub = x_max * (1.0 - (i * 0.01))
                    new_lb = x_min * (1.0 - (i * 0.01))
                    new_scale = (new_ub - new_lb) / (2**self.a_bit - 1)
                    new_zero_point = new_lb
                    x_q = (torch.round(
                        (x_detach - new_zero_point) / new_scale)).clamp(
                            self.a_min_q,
                            self.a_max_q) * new_scale + new_zero_point
                    score = lp_loss(x_detach, x_q, p=2.0, reduction='all')
                    if score < best_score:
                        best_score = score
                        res_scale = new_scale
                        # res_zero_point = new_zero_point
                self.a_scale.data.copy_(res_scale)
                self.a_zero_point.data.copy_(new_zero_point)
            self.a_scale.data.clamp_(self.eps)
            g = 1.0 / math.sqrt(x.numel() * self.a_max_q)
            cur_a_scale = grad_scale(self.a_scale, g)
            cur_a_zero_point = grad_scale(self.a_zero_point, g)
            x = round_pass((x - cur_a_zero_point) / cur_a_scale).clamp(
                self.a_min_q, self.a_max_q) * cur_a_scale + cur_a_zero_point
        if self.w_bit != 32:
            if self.training and self.w_init_state == 0:
                self.w_init_state.fill_(1)
                w_detach = weight.detach()
                if self.scale_channel == 1:
                    w_mean = w_detach.mean()
                    w_std = w_detach.std()
                else:
                    dim = [i for i in range(1, w_detach.dim())]
                    w_mean = w_detach.mean(dim=dim)
                    w_std = w_detach.std(dim=dim)
                v1 = torch.abs(w_mean - 3 * w_std)
                v2 = torch.abs(w_mean + 3 * w_std)
                w_scale.data.copy_(torch.max(v1, v2) / 2**(self.w_bit - 1))
            w_scale.data.clamp_(self.eps)
            g = 1.0 / math.sqrt(weight.numel() * self.w_max_q)
            cur_w_scale = grad_scale(w_scale, g)
            if self.scale_channel != 1:
                cur_w_scale = cur_w_scale.view(oup, 1)
            weight = round_pass(weight / cur_w_scale).clamp(
                self.w_min_q, self.w_max_q) * cur_w_scale

        out = F.linear(x, weight, bias)
        return out

    def to_static(self, x, oup=None):
        self.static = True
        inp = x.shape[1]
        if oup is None:
            oup = self.weight.shape[0]
        filters = self.weight[:oup, :inp].contiguous()
        bias = self.bias[:oup]
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(filters))
        self.register_parameter('bias_s', nn.Parameter(bias))
        if self.w_bit != 32:
            if self.scale_channel != 1:
                w_scale = self.w_scale[:oup]
            else:
                w_scale = self.w_scale
            self.w_scale.requires_grad = False
            self.register_parameter('w_scale_s', nn.Parameter(w_scale))
        return self.forward(x, oup)

    def get_flops_bitops(self, pre_sub_path):
        if tuple(pre_sub_path) not in self.ops_memory:
            pre_channels = pre_sub_path[0]
            bitops = 0.0
            flops = self.max_oup * (pre_channels + 1) // (
                64 / math.sqrt(self.w_bit * self.a_bit))
            self.ops_memory[tuple(pre_sub_path)] = (flops / 1e6, bitops / 1e6)
        return self.ops_memory[tuple(pre_sub_path)]


class DynamicBinConv2d(nn.Module):

    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups_list,
                 wh,
                 stride=1,
                 dilation=1):
        super().__init__()
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = list(set(ks_list))
        self.ks_list.sort()  # e.g., [3, 5, 7]
        self.max_ks = max(ks_list)
        self.groups_list = groups_list
        self.wh = wh
        self.stride = stride
        self.dilation = dilation
        shape = (self.max_oup, self.max_inp // min(groups_list), self.max_ks,
                 self.max_ks)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_buffer('scaling_factor', torch.ones(max_oup, 1, 1, 1))
        if len(self.ks_list) > 1:
            # register scaling parameters
            # matrix_7to5, matrix_5to3
            scale_params = {}
            for i in range(len(self.ks_list) - 1):
                ks_small = self.ks_list[i]
                ks_larger = self.ks_list[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params['matrix_%s' % param_name] = nn.Parameter(
                    torch.eye(ks_small**2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)
        if max(self.ks_list) != 1:
            name = 'fp2bin_%d' % max(self.ks_list)
            self.register_parameter(
                name, nn.Parameter(torch.eye(max(self.ks_list)**2)))
        self.reset_parameters()

        self.is_bin = True
        self.is_distill = False
        self.distill_loss_func = nn.MSELoss()
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_active_filter(self, weight, inp, oup, ks, groups):
        start, end = _sub_filter_start_end(self.max_ks, ks)
        filters = weight[:oup, :inp, start:end, start:end]
        if ks < self.max_ks:
            start_filter = weight[:oup, :inp, :, :]  # start with max kernel
            for i in range(len(self.ks_list) - 1, 0, -1):
                src_ks = self.ks_list[i]
                if src_ks <= ks:
                    break
                target_ks = self.ks_list[i - 1]
                start, end = _sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0),
                                                   _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    getattr(self, 'matrix_%dto%d' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1),
                                                   target_ks**2)
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1), target_ks,
                                                   target_ks)
                start_filter = _input_filter
            filters = start_filter
        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_inp = inp // groups
        sub_ratio = filters.size(1) // sub_inp
        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_inp
            filter_crops.append(sub_filter[:, start:start + sub_inp, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def fp2bin_filter(self, weight):
        oc, ic, ks, _ = weight.shape
        if ks > 1 and ks == max(self.ks_list):
            weight = weight.view(oc, ic, -1)
            weight = weight.view(-1, ks**2)
            weight = F.linear(
                weight,
                getattr(self, 'fp2bin_%d' % ks),
            )
            weight = weight.view(oc, ic, ks**2)
            weight = weight.view(oc, ic, ks, ks)
        return weight

    def forward(self, x, loss, sub_path):
        assert x.shape[-1] == self.wh
        oup, ks, groups = sub_path
        inp = x.shape[1]
        if oup == -1:
            oup = inp
        if self.static:
            weight = self.weight_s
            scaling_factor = self.scaling_factor_s
        else:
            weight = self.get_active_filter(self.weight, inp, oup, ks, groups)
            scaling_factor = self.scaling_factor[:oup]
            if self.is_bin:
                weight = self.fp2bin_filter(weight)
            weight = weight.contiguous()

        w_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        w_std = weight.std(dim=(1, 2, 3), keepdim=True)
        weight_n = (weight - w_mean) / w_std
        distill_weight = weight_n
        if self.is_bin:
            scaling_factor.copy_(
                torch.mean(abs(weight_n.data), dim=(1, 2, 3), keepdim=True))
            binary_weight = scaling_factor.detach() * SignFuncW.apply(weight_n)
        else:
            binary_weight = weight_n

        padding = (ks - 1) // 2 * self.dilation
        out = F.conv2d(x,
                       binary_weight,
                       None,
                       stride=self.stride,
                       padding=padding,
                       dilation=self.dilation,
                       groups=groups)
        if self.is_distill:
            binary_weight_n = binary_weight / binary_weight.abs().pow(2).mean(
                dim=(1, 2, 3), keepdim=True)
            with torch.no_grad():
                distill_weight_n = distill_weight / distill_weight.abs().pow(
                    2).mean(dim=(1, 2, 3), keepdim=True)
            loss += 0.1 * self.distill_loss_func(binary_weight_n,
                                                 distill_weight_n.detach())
        return out, loss

    def to_static(self, x, loss, sub_path):
        self.static = True
        assert x.shape[-1] == self.wh
        oup, ks, groups = sub_path
        inp = x.shape[1]
        if oup == -1:
            oup = inp
        weight = self.get_active_filter(self.weight, inp, oup, ks,
                                        groups).contiguous()
        if self.is_bin:
            weight = self.fp2bin_filter(weight).contiguous()
        self.register_parameter('weight_s', nn.Parameter(weight))
        self.register_buffer('scaling_factor_s', self.scaling_factor[:oup])
        self.weight.requires_grad = False
        if hasattr(self, 'matrix_7to5'):
            self.matrix_7to5.requires_grad = False
        if hasattr(self, 'matrix_5to3'):
            self.matrix_5to3.requires_grad = False
        return self.forward(x, loss, sub_path)


class DynamicPReLU(nn.Module):

    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.prelu = nn.PReLU(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        if self.static:
            out = self.prelu_s(x)
        else:
            out = F.prelu(x, self.prelu.weight[:feature_dim])
        return out

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        weight = self.prelu.weight[:feature_dim]
        self.prelu.weight.requires_grad = False
        self.prelu_s = nn.PReLU(feature_dim)
        self.prelu_s.eval()
        self.prelu_s.weight.data.copy_(weight.data)
        return self.forward(x)


class DynamicBatchNorm2d(nn.Module):

    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.bn = nn.BatchNorm2d(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        if self.static:
            bn = self.bn_s
        else:
            bn = self.bn
        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum

        if bn.training and bn.track_running_stats:
            if bn.num_batches_tracked is not None:  # type: ignore[has-type]
                bn.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = bn.momentum

        if bn.training:
            bn_training = True
        else:
            bn_training = (bn.running_mean is None) and (bn.running_var is
                                                         None)
        if self.static:
            return F.batch_norm(
                x,
                bn.running_mean
                if not bn.training or bn.track_running_stats else None,
                bn.running_var
                if not bn.training or bn.track_running_stats else None,
                bn.weight,
                bn.bias,
                bn_training,
                exponential_average_factor,
                bn.eps,
            )
        else:
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim]
                if not bn.training or bn.track_running_stats else None,
                bn.running_var[:feature_dim]
                if not bn.training or bn.track_running_stats else None,
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn_training,
                exponential_average_factor,
                bn.eps,
            )

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        running_mean = self.bn.running_mean[:feature_dim]
        running_var = self.bn.running_var[:feature_dim]
        weight = self.bn.weight[:feature_dim]
        bias = self.bn.bias[:feature_dim]
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False
        self.bn_s = nn.BatchNorm2d(feature_dim)
        self.bn_s.eval()
        self.bn_s.running_mean.copy_(running_mean)
        self.bn_s.running_var.copy_(running_var)
        self.bn_s.weight.data.copy_(weight.data)
        self.bn_s.bias.data.copy_(bias.data)
        return self.forward(x)
