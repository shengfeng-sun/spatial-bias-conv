class Sp_conv(nn.Module):
    def __init__(self, input_channels=32, scale=4):
        super(Sp_conv, self).__init__()
        self.ceil = input_channels // scale
        self.rs_channels = scale * (self.ceil - 1)
        self.sp_channels = scale * 2
        sp_channels = self.sp_channels
        rs_channels = self.rs_channels
        assert sp_channels + rs_channels - scale == input_channels
        self.channel_Reduction = nn.Conv2d(input_channels, sp_channels, 1)
        self.rs_conv = nn.Conv2d(input_channels, rs_channels, 1)
        self.avg_pool = nn.AvgPool2d((2, 2))
        self.flat = nn.Flatten(1, 3)
        # self.conv1x2 = nn.Conv2d(sp_channels,sp_channels//2,(1,7),stride=4,padding=[0,1])
        self.conv1x2 = nn.Conv2d(sp_channels, sp_channels // 2, (1, 3), padding=[0, 1])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = x
        residual = self.rs_conv(residual)
        x = self.channel_Reduction(x)
        B, C, H, W = x.shape
        x = self.avg_pool(x)

        x = einops.rearrange(x, 'B C H W -> B C 1 (H W)')
        x = self.conv1x2(x)
        x = x.squeeze(2).view(B, math.ceil(C / 2) // 1, math.ceil(H / 2) // 1, math.ceil(W / 2) // 1)

        x = self.up(x)

        x = torch.cat([x, residual], dim=1)

        return x
