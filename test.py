import argparse
from pathlib import Path

# 追加モジュール
import numpy as np
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

import utils.dwt as dwt
import utils.dct as dct
import utils.fft as fft


parser = argparse.ArgumentParser(
    prog='Test DWT DCT FFT',
    description='各画像の周波数分解を行った画像を作成します'
)

parser.add_argument(
    '-np', '--num_progress', help='何層分周波数分解するかを指定します。(5~0)',
    type=int, default=5
)

# 入力に関するコマンドライン引数
parser.add_argument(
    '-id', '--input_dir', help='入力ディレクトリの名前を指定します。',
    type=str, default=None,
)

#   モデルの保存
parser.add_argument(
    '--dct', help='dctにて周波数分解を実行します。',
    action='store_true'
)

parser.add_argument(
    '--fft', help='fftにて周波数分解を実行します。',
    action='store_true'
)

# コマンドライン引数をパースする
args = parser.parse_args()
num_progress = args.num_progress
num_samples = 1
device = 'cuda'

if args.dct:
    w_type = 'dct'
    wavelet = dct.DCT(num_progress)
elif args.fft:
    w_type = 'fft'
    wavelet = fft.FFT()
else:
    w_type = 'dwt'
    wavelet = dwt.DWT()
wavelet = wavelet.to(device)

if args.input_dir is not None:
    INPUT_DIR = Path(f'{args.input_dir}')


image_path = INPUT_DIR
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
image = transform(image)
image = torch.unsqueeze(image, 0)
image = image.to(device)
sample_images = wavelet(image, num_progress)

OUTPUT_DIR = Path('./image/sample')
sample_dir = OUTPUT_DIR.joinpath(f'{w_type}')
with torch.no_grad():
    vutils.save_image(
        sample_images,
        sample_dir.joinpath(f'{w_type}_{num_progress}.png'),
        nrow=int(np.sqrt(num_samples)),
        range=(-1.0, 1.0),
        normalize=True
    )
