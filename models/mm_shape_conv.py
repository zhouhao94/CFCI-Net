# coding=utf-8

import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class ShapeConvFB(Module):
    """
       ShapeConv2d can be used as an alternative for torch.nn.Conv2d.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ShapeConvFB, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.testing = not self.training
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.weight1 = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        self.weight2 = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

        if M * N > 1:
            self.Shape1 = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            self.Base1 = Parameter(torch.Tensor(1))
            init_zero1 = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)

            init_one1 = np.ones([1], dtype=np.float32)
            self.Shape1.data = torch.from_numpy(init_zero1)
            self.Base1.data = torch.from_numpy(init_one1)

            eye1 = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag1 = eye1.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros1 = torch.zeros([1, M * N, self.D_mul % (M * N)])
                self.D_diag1 = Parameter(torch.cat([D_diag1, zeros1], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag1 = Parameter(D_diag1, requires_grad=False)


            self.Shape2 = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            self.Base2 = Parameter(torch.Tensor(1))
            init_zero2 = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)

            init_one2 = np.ones([1], dtype=np.float32)
            self.Shape2.data = torch.from_numpy(init_zero2)
            self.Base2.data = torch.from_numpy(init_one2)

            eye2 = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag2 = eye2.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros2 = torch.zeros([1, M * N, self.D_mul % (M * N)])
                self.D_diag2 = Parameter(torch.cat([D_diag2, zeros2], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag2 = Parameter(D_diag2, requires_grad=False)

        if bias:
            self.bias1 = Parameter(torch.Tensor(out_channels))
            fan_in1, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1)
            init.uniform_(self.bias1, -bound1, bound1)

            self.bias2 = Parameter(torch.Tensor(out_channels))
            fan_in2, _ = init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2)
            init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.register_parameter('bias1', None)

            self.register_parameter('bias2', None)

        self.control_params1 = Parameter(torch.ones(2)) #Parameter(torch.ones(2, out_channels))
        self.control_params2 = Parameter(torch.ones(2)) #Parameter(torch.ones(2, out_channels))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias1 is None:
            s += ', bias1=False'
        if self.bias2 is None:
            s += ', bias2=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(ShapeConvFB, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def compute_shape_w1(self, alpha=0.5, beta=0.5):
        # (input_channels, D_mul, M * N)
        Shape = self.Shape1 + self.D_diag1  # (1, M * N, self.D_mul)
        Base = self.Base1
        W = torch.reshape(self.weight1, (self.out_channels // self.groups, self.in_channels, self.D_mul))
        W_base = torch.mean(W, [2], keepdims=True)  # (self.out_channels // self.groups, self.in_channels)
        W_shape = W - W_base  # (self.out_channels // self.groups, self.in_channels, self.D_mul)

        # einsum outputs (out_channels // groups, in_channels, M * N),
        # which is reshaped to
        # (out_channels, in_channels // groups, M, N)
        D_shape = torch.reshape(torch.einsum('ims,ois->oim', Shape, W_shape), self.weight1.shape)
        D_base = torch.reshape(W_base * Base, (self.out_channels, self.in_channels // self.groups, 1, 1))

        #alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #beta = beta.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        DW = alpha*D_shape + beta*D_base
        
        return DW
    
    def compute_shape_w2(self, alpha=0.5, beta=0.5):
        # (input_channels, D_mul, M * N)
        Shape = self.Shape2 + self.D_diag2  # (1, M * N, self.D_mul)
        Base = self.Base2
        W = torch.reshape(self.weight2, (self.out_channels // self.groups, self.in_channels, self.D_mul))
        W_base = torch.mean(W, [2], keepdims=True)  # (self.out_channels // self.groups, self.in_channels)
        W_shape = W - W_base  # (self.out_channels // self.groups, self.in_channels, self.D_mul)

        # einsum outputs (out_channels // groups, in_channels, M * N),
        # which is reshaped to
        # (out_channels, in_channels // groups, M, N)
        D_shape = torch.reshape(torch.einsum('ims,ois->oim', Shape, W_shape), self.weight2.shape)
        D_base = torch.reshape(W_base * Base, (self.out_channels, self.in_channels // self.groups, 1, 1))

        #alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #beta = beta.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        DW = alpha*D_shape + beta*D_base
        
        return DW

    def forward(self, input1, input2):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        if M * N > 1 and not self.testing:  # train and val
            param1 = F.softmax(self.control_params1, dim=0)
            param2 = F.softmax(self.control_params2, dim=0)
            DW1 = self.compute_shape_w1(param1[0], param2[0])
            DW2 = self.compute_shape_w2(param1[1], param2[1])
        else:   # test
            DW1 = self.weight1
            DW2 = self.weight2

        return self._conv_forward(input1, DW1, self.bias1), self._conv_forward(input2, DW2, self.bias2)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.testing = not self.training
        super(ShapeConvFB, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                       missing_keys, unexpected_keys, error_msgs)
        if self.kernel_size[0] * self.kernel_size[1] > 1 and not self.training:
            param1 = F.softmax(self.control_params1, dim=0)
            param2 = F.softmax(self.control_params2, dim=0)

            self.weight1.data = self.compute_shape_w1(param1[0], param2[0])
            self.weight2.data = self.compute_shape_w2(param1[1], param2[1])


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)