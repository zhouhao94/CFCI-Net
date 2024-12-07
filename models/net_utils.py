import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from timm.models.layers import trunc_normal_
from torch.nn.parameter import Parameter
from .sam import SAM
from .mm_shape_conv import ShapeConvFB

# Feature Rectify Module
class SelfCalibration(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SelfCalibration, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // reduction, self.dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, _, H, W = x.shape

        avg = self.avg_pool(x).view(B, self.dim)
        max = self.max_pool(x).view(B, self.dim)

        x_avg = self.mlp(avg).view(B, self.dim, 1, 1)
        x_max = self.mlp(max).view(B, self.dim, 1, 1)

        x_weights = self.sigmoid(x_avg+x_max)
        x_out = x * x_weights

        return x_out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class GlobalCalibration(nn.Module):
    def __init__(self, dim, reduction=1):
        super(GlobalCalibration, self).__init__()
        self.dim = dim
        self.channel_pool_x1 = ChannelPool()
        self.channel_pool_x2 = ChannelPool()
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(4, dim//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(dim//reduction, dim//reduction, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim//reduction, 2, kernel_size=1, bias=True),
                        nn.Sigmoid()
                        )

    def forward(self, x1, x2):
        x1_pool = self.channel_pool_x1(x1) # Bx2xHxW
        x2_pool = self.channel_pool_x2(x2) # Bx2xHxW

        x_hidden = torch.cat((x1_pool, x2_pool), dim=1) # Bx4xHxW
        global_weights = self.channel_embed(x_hidden) # Bx2xHxW
        x1_weights, x2_weights = torch.split(global_weights, 1, dim=1) # Bx1xHxW

        x1_out = x1 * x1_weights
        x2_out = x2 * x2_weights

        return x1_out, x2_out

'''
class SAN(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(SAN, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(in_planes)
        self.bn1_2 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, y):
        out1 = self.relu(self.bn1_1(x))
        out2 = self.relu(self.bn1_2(y))
        out = self.relu(self.bn2(self.sam(out1,out2)))
        out = self.conv(out)
        return out
'''

class LocalCalibration(nn.Module):
    def __init__(self, sa_type, dim, reduction, kernel_size=7, stride=1):
        super(LocalCalibration, self).__init__()
        self.x1_calibration = SAM(sa_type, dim, dim//reduction, dim, 8, kernel_size, stride)
        self.x2_calibration = SAM(sa_type, dim, dim//reduction, dim, 8, kernel_size, stride)
    
    def forward(self, x1, x2):
        x1_out = self.x1_calibration(x1, x2)
        x2_out = self.x2_calibration(x2, x1)

        return x1_out, x2_out

class FeatureCalibrationModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_g=.5, lambda_l=.5, sa_type=0, kernel_size=7, stride=1):
        super(FeatureCalibrationModule, self).__init__()
        self.lambda_g = lambda_g
        self.lambda_l = lambda_l
        self.self_calibration_x1 = SelfCalibration(dim=dim, reduction=reduction)
        self.self_calibration_x2 = SelfCalibration(dim=dim, reduction=reduction)
        self.global_calibration = GlobalCalibration(dim=dim, reduction=reduction)
        self.local_calibration = LocalCalibration(sa_type, dim, reduction, kernel_size, stride)

        self.control_params1 = Parameter(torch.ones(2))
        self.control_params2 = Parameter(torch.ones(2))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        s_x1 = self.self_calibration_x1(x1)
        s_x2 = self.self_calibration_x2(x2)

        g_x1, g_x2 = self.global_calibration(s_x1, s_x2)
        l_x1, l_x2 = self.local_calibration(s_x1, s_x2)
        
        param1 = F.softmax(self.control_params1, dim=0)
        param2 = F.softmax(self.control_params2, dim=0)

        out_x1 = x1 + param1[0] * g_x2 + param1[1] * l_x2
        out_x2 = x2 + param2[0] * g_x1 + param2[1] * l_x1

        return out_x1, out_x2
        

class FeatureFusion(nn.Module):
    def __init__(self, dim, reduction=1):
        super(FeatureFusion, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(4*dim, 4*dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(4*dim // reduction, 2*dim))
    
    def forward(self, x1, x2):
        B, _, H, W = x1.shape

        x1_avg = self.avg_pool(x1).view(B, self.dim)
        x1_max = self.max_pool(x1).view(B, self.dim)

        x2_avg = self.avg_pool(x2).view(B, self.dim)
        x2_max = self.max_pool(x2).view(B, self.dim)

        x_hidden = torch.cat((x1_avg, x1_max, x2_avg, x2_max), dim=-1)
        x_hidden = self.mlp(x_hidden).reshape(B, self.dim, 2)

        x_weights = F.softmax(x_hidden, dim=-1) # B, C, 2
        x1_weights, x2_weights = torch.split(x_weights, 1, dim=-1) # Bx1xHxW

        x1_weights = x1_weights.view(B, self.dim, 1, 1)
        x2_weights = x2_weights.view(B, self.dim, 1, 1)

        x_fusion = x1_weights*x1 + x2_weights*x2

        return x_fusion


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureIntegrationModule(nn.Module):
    def __init__(self, dim, kernel_size, reduction=1, norm_layer=nn.BatchNorm2d, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(FeatureIntegrationModule, self).__init__()
        self.feature_balance = ShapeConvFB(dim, dim, kernel_size, D_mul, stride,
                 padding, dilation, groups, bias, padding_mode)
        self.feature_fusion_bs = FeatureFusion(dim, reduction)
        self.feature_fusion_fu = FeatureFusion(dim, reduction)
        self.channel_emb = ResBlock(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        bs_x1, bs_x2 = self.feature_balance(x1, x2)
        x_bs = self.feature_fusion_bs(bs_x1, bs_x2)
        x_fu = self.feature_fusion_fu(x1, x2)

        x_hidden = torch.cat((x_bs, x_fu), dim=1)

        x_out = self.channel_emb(x_hidden)

        return x_out