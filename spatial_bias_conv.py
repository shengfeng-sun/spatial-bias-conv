import torch
import torch.nn as nn
from einops import einops
from typing import Dict
from torchinfo import summary
class Sp_conv(nn.Module):
    def __init__(self,input_channels=3):
        C1= input_channels // 2
        super(Sp_conv, self).__init__()
        self.channel_Reduction = nn.Conv2d(input_channels,C1,1)
        self.avg_pool = nn.AvgPool2d((2,2))
        self.flat = nn.Flatten(1,3)
        self.conv1x2 = nn.Conv2d(C1,C1//2,(1,7),stride=4,padding=[0,2])
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(C1*2 + C1//2)
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # X : 4, 3, 224,224
        residual = x
        x = self.channel_Reduction(x)
        B, C, H, W = x.shape
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'B C H W -> B C 1 (H W)')
        x = self.conv1x2(x)

        x = x.squeeze(2).view(B,C//2,H//4,W//4)

        x = self.up(x)
        x = torch.cat([x,residual],dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return {"out": x}


if __name__ == '__main__':
    model = Sp_conv(input_channels=32)
    summary(model, (1, 32, 224, 224))
