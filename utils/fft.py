import torch.fft
import torch.nn as nn
from math import ceil


class FFT(nn.Module):
    def forward(self, images: torch.Tensor, k: int = 1) -> torch.Tensor:
        images_shifted = (images + 1) / 2
        coefs = torch.fft.fft(images_shifted)  # 便宜上(B, C, H, W)とする
        lpf = torch.zeros(coefs.size(2), coefs.size(3), device='cuda')
        mid = ceil(lpf.size(1) / 2)
        lpf_list = [16/16, 12/16, 8/16, 4/16, 2/16, 1/16]
        lpf_ratio = lpf_list[k] * mid
        low = ceil(mid - lpf_ratio)
        hi = ceil(mid + lpf_ratio)
        lpf[low: hi, low: hi] = 1
        # print(lpf)
        lpf_dc = torch.cat([
            lpf[mid:, mid:],
            lpf[mid:, :mid]
            ], dim=1)
        lpf_ba = torch.cat([
            lpf[:mid, mid:],
            lpf[:mid, :mid]
            ], dim=1)
        lpf_shifted = torch.cat([
            lpf_dc, lpf_ba
            ], dim=0)
        # print(lpf_shifted)
        adaptted = coefs * lpf_shifted
        ifft_images = torch.fft.ifft(adaptted)
        rgb = torch.real(ifft_images) * 2 - 1
        return rgb
