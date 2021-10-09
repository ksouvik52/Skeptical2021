"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.relu(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.relu(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.relu(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

# def dowmsampleBottleneck(channel_in, channel_out, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(),
#         nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
#         nn.BatchNorm2d(channel_out),
#         nn.ReLU(),
#         )



class MobileNetV2(nn.Module):

    def __init__(self, class_num=10):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        #insert scala layers

        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.scala1 = nn.Sequential(

            SepConv(
                channel_in=24,
                channel_out=64,stride=2
            ),

            SepConv(
                channel_in=64,
                channel_out=160,
            ),

            SepConv(
                channel_in=160,
                channel_out=320,
            ),
            SepConv(
                channel_in=320,
                channel_out=640,
            ),
            SepConv(
                channel_in=640,
                channel_out=1280,
            ),
            nn.AvgPool2d(5, 5)
        )

        self.scala2 = nn.Sequential(

            SepConv(
                channel_in=64,
                channel_out=160,
            ),

            SepConv(
                channel_in=160,
                channel_out=320,
            ),
            SepConv(
                channel_in=320,
                channel_out=640,
            ),
            SepConv(
                channel_in=640,
                channel_out=1280,
            ),
            nn.AvgPool2d(5, 5)
        )

        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=160,
                channel_out=320,
            ),
            SepConv(
                channel_in=320,
                channel_out=640,
            ),
            SepConv(
                channel_in=640,
                channel_out=1280,
            ),
            nn.AvgPool2d(5, 5)
        )

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=24,
                channel_out=24
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=64,
                channel_out=64
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=160,
                channel_out=160
            ),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Sigmoid()
        )

        self.conv_1out = nn.Conv2d(1280, class_num, 1)
        self.conv_2out = nn.Conv2d(1280, class_num, 1)
        self.conv_3out = nn.Conv2d(1280, class_num, 1)
        self.conv_4out = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)#16,34,34
        x = self.stage2(x)#24,17,17

        fea1 = self.attention1(x)
        fea1 = fea1 * x
        out1_feature = self.scala1(fea1)
        out1 = self.conv_1out(out1_feature)
        out1 = out1.view(out1.size(0),-1)

        x = self.stage3(x)#32,9,9
        x = self.stage4(x)#64,5,5

        fea2 = self.attention2(x)
        fea2 = fea2 * x
        out2_feature = self.scala2(fea2)
        out2 = self.conv_2out(out2_feature)
        out2 = out2.view(out2.size(0),-1)

        x = self.stage5(x)#94,5,5
        x = self.stage6(x)#160,5,5

        fea3 = self.attention3(x)
        fea3 = fea3 * x
        out3_feature = self.scala3(fea3)
        out3 = self.conv_3out(out3_feature)
        out3 = out3.view(out3.size(0), -1)

        x = self.stage7(x)#320,5,5
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        out4_feature = x
        x = self.conv_4out(x)
        out4 = x.view(x.size(0), -1)

        return [out4,out3,out2,out1], [out4_feature,out3_feature,out2_feature,out1_feature]

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def MobileNetV2_self(**kwargs):
    return MobileNetV2(**kwargs)

if __name__ == "__main__":
    model = mobilenetv2_self(class_num=100)
    # model.cuda()
    x = torch.randn(64,3,32,32)
    y = model(x)
