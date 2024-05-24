import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2


class FeatureReinforcementModule(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(FeatureReinforcementModule, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 1
        self.conv_scale1_c1 = nn.Sequential(
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16, stride=16),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # scale 2
        self.conv_scale1_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # scale 3
        self.conv_scale1_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # scale 4
        self.conv_scale1_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # scale 5
        self.conv_scale1_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        # fusion
        self.conv_aggregation_s1 = FeatureFusionModule(self.mid_d * 5, self.in_d[0], self.out_d)
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 5, self.in_d[1], self.out_d)
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 5, self.in_d[2], self.out_d)
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 5, self.in_d[3], self.out_d)
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 5, self.in_d[4], self.out_d)

    def forward(self, c1, c2, c3, c4, c5):
        # scale 1
        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)
        c1_s5 = self.conv_scale5_c1(c1)

        # scale 2
        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=(2, 2), mode='bilinear')
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)
        c2_s5 = self.conv_scale5_c2(c2)

        # scale 3
        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=(4, 4), mode='bilinear')
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)
        c3_s5 = self.conv_scale5_c3(c3)

        # scale 4
        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=(8, 8), mode='bilinear')
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=(4, 4), mode='bilinear')
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        c4_s4 = self.conv_scale4_c4(c4)
        c4_s5 = self.conv_scale5_c4(c4)

        # scale 5
        c5_s1 = F.interpolate(self.conv_scale1_c5(c5), scale_factor=(16, 16), mode='bilinear')
        c5_s2 = F.interpolate(self.conv_scale2_c5(c5), scale_factor=(8, 8), mode='bilinear')
        c5_s3 = F.interpolate(self.conv_scale3_c5(c5), scale_factor=(4, 4), mode='bilinear')
        c5_s4 = F.interpolate(self.conv_scale4_c5(c5), scale_factor=(2, 2), mode='bilinear')
        c5_s5 = self.conv_scale5_c5(c5)

        s1 = self.conv_aggregation_s1(torch.cat([c1_s1, c2_s1, c3_s1, c4_s1, c5_s1], dim=1), c1)
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2, c2_s2, c3_s2, c4_s2, c5_s2], dim=1), c2)
        s3 = self.conv_aggregation_s3(torch.cat([c1_s3, c2_s3, c3_s3, c4_s3, c5_s3], dim=1), c3)
        s4 = self.conv_aggregation_s4(torch.cat([c1_s4, c2_s4, c3_s4, c4_s4, c5_s4], dim=1), c4)
        s5 = self.conv_aggregation_s5(torch.cat([c1_s5, c2_s5, c3_s5, c4_s5, c5_s5], dim=1), c5)
        return s1, s2, s3, s4, s5


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=3, stride=1, padding=1, groups=self.fuse_d),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out


class CrossConcat(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.diff = nn.Sequential(
            nn.Conv2d(in_d * 2, in_d, kernel_size=3, padding=1, stride=1, groups=in_d),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d, in_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        b, c, h, w = x1.shape
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(b, -1, h, w)
        x = self.diff(x)
        return x


class GroupFusion(nn.Module):
    def __init__(self, in_d, out_d):
        super(GroupFusion, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = self.conv(x)
        return x


class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(TemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.gf_x1 = GroupFusion(self.in_d, self.out_d)
        self.gf_x2 = GroupFusion(self.in_d, self.out_d)
        self.gf_x3 = GroupFusion(self.in_d, self.out_d)
        self.gf_x4 = GroupFusion(self.in_d, self.out_d)
        self.gf_x5 = GroupFusion(self.in_d, self.out_d)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5):
        # temporal fusion
        c1 = self.gf_x1(x1_1, x2_1)
        c2 = self.gf_x2(x1_2, x2_2)
        c3 = self.gf_x3(x1_3, x2_3)
        c4 = self.gf_x4(x1_4, x2_4)
        c5 = self.gf_x5(x1_5, x2_5)

        return c1, c2, c3, c4, c5


class GlobalContextAggregation(nn.Module):
    def __init__(self, in_d=64, out_d=64, reduction=2, group=4):
        super(GlobalContextAggregation, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.mid_d = out_d * reduction
        self.group = group
        assert self.mid_d % self.group == 0, "fail to split groups"
        self.split_d = self.mid_d // group
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Conv2d(self.in_d, self.mid_d, kernel_size=1, stride=1)
        self.conv_list = nn.ModuleList()
        for i in range(group):
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(self.split_d, self.split_d, kernel_size=3, stride=1, padding=i + 1, dilation=i + 1,
                              groups=self.split_d),
                    nn.BatchNorm2d(self.split_d),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x4, x5):
        x = self.conv3x3(x4 + F.interpolate(x5, scale_factor=(2, 2), mode="bilinear"))
        b, c, h, w = x.size()
        x = self.conv1x1(x)
        x = x.view(b, self.group, self.split_d, h, w)  # bs,s,ci,h,w
        for idx, conv in enumerate(self.conv_list):
            x[:, idx, :, :, :] = self.conv_list[idx](x[:, idx, :, :, :])
        x = x.view(b, -1, h, w)
        return self.out_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, mid_d):
        super(DecoderBlock, self).__init__()
        self.mid_d = mid_d
        self.conv_high = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1)
        self.conv_global = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, x_low, x_high, global_context):
        batch, channels, height, width = x_low.shape
        x_high = F.interpolate(self.conv_high(x_high), size=(height, width), mode="bilinear")
        global_context = F.interpolate(self.conv_global(global_context), size=(height, width), mode="bilinear")
        x_low = self.fusion(x_low + x_high + global_context)
        mask = self.cls(x_low)
        return x_low, mask


class ChannelReferenceAttention(nn.Module):
    def __init__(self, in_d):
        super(ChannelReferenceAttention, self).__init__()
        self.in_d = in_d
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1)
        )
        self.high_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1)
        )
        self.low_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_context, high_context, global_context):
        b, c, h, w = low_context.shape
        low_context_pool = self.low_conv(low_context)
        high_context_pool = self.high_conv(high_context)
        global_context_pool = self.global_conv(global_context)
        low_context_pool = low_context_pool.squeeze(dim=-1)
        high_context_pool = high_context_pool.squeeze(dim=-1).permute(0, 2, 1)
        global_context_pool = global_context_pool.squeeze(dim=-1).permute(0, 2, 1)
        att_l_h = torch.bmm(low_context_pool, high_context_pool)
        att_l_g = torch.bmm(low_context_pool, global_context_pool)
        att = torch.sigmoid(att_l_h + att_l_g)
        out = torch.bmm(att, low_context.view(b, c, -1))
        out = self.out_conv(out.view(b, c, h, w)) + low_context
        return out


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        self.cra4 = ChannelReferenceAttention(self.mid_d)
        self.cra3 = ChannelReferenceAttention(self.mid_d)
        self.cra2 = ChannelReferenceAttention(self.mid_d)
        self.cra1 = ChannelReferenceAttention(self.mid_d)
        self.db_p4 = DecoderBlock(self.mid_d)
        self.db_p3 = DecoderBlock(self.mid_d)
        self.db_p2 = DecoderBlock(self.mid_d)
        self.db_p1 = DecoderBlock(self.mid_d)
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_p5 = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1)

    def forward(self, d1, d2, d3, d4, d5, gc_d4):
        # high-level
        p4 = self.cra4(d4, d5, gc_d4)
        p4, mask_p4 = self.db_p4(p4, d5, gc_d4)
        p3 = self.cra3(d3, p4, gc_d4)
        p3, mask_p3 = self.db_p3(p3, p4, gc_d4)
        p2 = self.cra2(d2, p3, gc_d4)
        p2, mask_p2 = self.db_p2(p2, p3, gc_d4)
        p1 = self.cra1(d1, p2, gc_d4)
        p1, mask_p1 = self.db_p1(p1, p2, gc_d4)
        return mask_p1, mask_p2, mask_p3, mask_p4


class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.frm = FeatureReinforcementModule(channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.gca = GlobalContextAggregation(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.frm(x1_1, x1_2, x1_3, x1_4, x1_5)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.frm(x2_1, x2_2, x2_3, x2_4, x2_5)
        # temporal fusion
        c1, c2, c3, c4, c5 = self.tfm(x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5)
        # global context of high-level and low-level
        gc_c4 = self.gca(c4, c5)

        # fpn
        mask_p1, mask_p2, mask_p3, mask_p4 = self.decoder(c1, c2, c3, c4, c5, gc_c4)

        # change map
        mask_p1 = F.interpolate(mask_p1, scale_factor=(2, 2), mode='bilinear')
        mask_p1 = torch.sigmoid(mask_p1)
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)

        return mask_p1, mask_p2, mask_p3, mask_p4
