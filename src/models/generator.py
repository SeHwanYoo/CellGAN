import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import InitLayer, UpBlock, SN_Conv2d, EmbeddingNetwork, SGCBlock


class Generator(nn.Module):
    def __init__(self, nc=3, nz=128, ngf=32, n_class=5, d_cond=128, n_layer=4):
        super().__init__()

        # Feature channels at each spatial resolution
        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 8, 64: 4, 128: 2, 256: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        # Network structure
        self.cls_map_net = EmbeddingNetwork(n_class, d_cond, n_layer)
        
        self.init_z = InitLayer(nz, channel=nfc[4])
        self.up_8 = UpBlock(nfc[4], nfc[8], d_cond)
        self.up_16 = UpBlock(nfc[8], nfc[16], d_cond)
        self.up_32 = UpBlock(nfc[16], nfc[32], d_cond)
        self.up_64 = UpBlock(nfc[32], nfc[64], d_cond)
        self.up_128 = UpBlock(nfc[64], nfc[128], d_cond)
        self.up_256 = UpBlock(nfc[128], nfc[256], d_cond)

        self.se_8_64 = SGCBlock(nfc[8], nfc[64])
        self.se_16_128 = SGCBlock(nfc[16], nfc[128])
        self.se_32_256 = SGCBlock(nfc[32], nfc[256])

        self.to_out = nn.Sequential(
            SN_Conv2d(nfc[256], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def class_embed(self, label):
        return self.cls_map_net(label)

    def gennerate(self, z, cond):
        feat_4 = self.init_z(z)
        feat_8 = self.up_8(feat_4, cond)
        feat_16 = self.up_16(feat_8, cond)
        feat_32 = self.up_32(feat_16, cond)
        feat_64 = self.se_8_64(feat_8, self.up_64(feat_32, cond))
        feat_128 = self.se_16_128(feat_16, self.up_128(feat_64, cond))
        feat_256 = self.se_32_256(feat_32, self.up_256(feat_128, cond))
        img = self.to_out(feat_256)

        return img
    
    def fft_forward(self, x):
        B, C, H, W = x.size()
        
        # FFT: Moving to the frequency domain
        fft_x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        
        # Splitting into real and imaginary components for separate manipulation
        fft_real = fft_x.real
        fft_imag = fft_x.imag
        
        # Concatenating real and imaginary parts
        fft_x_combined = torch.cat([fft_real, fft_imag], dim=1)
        
        # Applying convolution in frequency domain
        fft_x_conv = F.relu(self.conv3(fft_x_combined))
        
        # Splitting back into real and imaginary parts
        fft_real, fft_imag = torch.chunk(fft_x_conv, 2, dim=1)
        
        # Reconstructing the complex tensor and applying inverse FFT to return to image domain
        fft_x = torch.complex(fft_real, fft_imag)
        x_out = torch.fft.ifft2(fft_x, dim=(-2, -1), norm='ortho')
        
        return x_out.real  # Output should be real-valued

    def forward(self, z, label, x=None):
        cond = self.cls_map_net(label)
        img_out =self.gennerate(z, cond)

        # if x is not None:
        #     x_fft = self.fft_forward(x)
        #     img_out = img_out + x_fft

        return img_out
