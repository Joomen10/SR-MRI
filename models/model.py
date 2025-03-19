import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvComplex(nn.Module):
    """
    A double convolution block with Batch Normalization.
    Optionally, you can add dropout if needed.
    """
    def __init__(self, in_channels, out_channels, dropout=False, p=0.5):
        super(DoubleConvComplex, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.insert(4, nn.Dropout2d(p))
        self.double_conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.double_conv(x)


class UNet2DDeep(nn.Module):
    """
    A deeper, more complex 2D U-Net model with 5 down/up sampling levels.
    Dynamically pads input so each dimension is a multiple of 32,
    preventing shape mismatches in skip connections.
    
    Architecture (channels shown in parentheses):
      -- Encoder --
        inc:   in_channels -> 64
        down1: 64         -> 128
        down2: 128        -> 256
        down3: 256        -> 512
        down4: 512        -> 1024
        down5: 1024       -> 2048 (bottleneck, with dropout)
        
      -- Decoder --
        up1:   2048 -> 1024,  concat with skip (1024) => DoubleConv -> 1024
        up2:   1024 -> 512,   concat with skip (512)  => DoubleConv -> 512
        up3:   512  -> 256,   concat with skip (256)  => DoubleConv -> 256
        up4:   256  -> 128,   concat with skip (128)  => DoubleConv -> 128
        up5:   128  -> 64,    concat with skip (64)   => DoubleConv -> 64
        
      -- Output --
        outc: 64 -> out_channels
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2DDeep, self).__init__()
        
        # ----------------
        #   Encoder
        # ----------------
        self.inc = DoubleConvComplex(in_channels, 64)
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvComplex(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvComplex(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvComplex(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvComplex(512, 1024)
        )
        # Bottleneck with dropout
        self.down5 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvComplex(1024, 2048, dropout=True, p=0.5)
        )
        
        # ----------------
        #   Decoder
        # ----------------
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvComplex(2048, 1024)
        
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvComplex(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConvComplex(512, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConvComplex(256, 128)
        
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up5 = DoubleConvComplex(128, 64)
        
        # Final output conv
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Pads x so (H, W) are multiples of 32, runs the 5-level U-Net,
        then slices the output back to the original size.
        """
        # Original input shape
        B, C, H, W = x.shape
        
        # 1) Compute how much padding is needed so H and W are multiples of 32
        def _pad_length(dim_size):
            remainder = dim_size % 32
            return (32 - remainder) if remainder != 0 else 0

        pad_h = _pad_length(H)
        pad_w = _pad_length(W)
        
        # 2) Pad on bottom/right side if needed
        # F.pad format: (left, right, top, bottom)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Shapes after padding
        _, _, Hpad, Wpad = x.shape
        
        # ----------------
        #   Encoder
        # ----------------
        x1 = self.inc(x)          
        x2 = self.down1(x1)       
        x3 = self.down2(x2)       
        x4 = self.down3(x3)       
        x5 = self.down4(x4)       
        x6 = self.down5(x5)       

        # ----------------
        #   Decoder
        # ----------------
        # Level 1 up
        x = self.up1(x6)          
        x = torch.cat([x, x5], dim=1)  
        x = self.conv_up1(x)      

        # Level 2 up
        x = self.up2(x)           
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up2(x)      

        # Level 3 up
        x = self.up3(x)           
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up3(x)      

        # Level 4 up
        x = self.up4(x)           
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up4(x)      

        # Level 5 up
        x = self.up5(x)           
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up5(x)      

        # Final 1×1 conv
        x = self.outc(x)          

        # 3) Un-pad the result to the original (H, W)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]   # slice off the extra padding

        return x
