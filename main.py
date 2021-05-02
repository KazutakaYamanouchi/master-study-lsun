# 標準モジュール
import argparse
import csv
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)

from pathlib import Path
import random
import sys
from time import perf_counter

# 追加モジュール
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# 自作モジュール
from models.generator import Generator
from models.discriminator import Discriminator
import utils.dwt as dwt
import utils.dct as dct
import utils.fft as fft

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# 訓練に関する引数
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=100, metavar='B'
)
parser.add_argument(
    '-e', '--num-epochs', help='学習エポック数を指定します。',
    type=int, default=50, metavar='E'
)

parser.add_argument(
    '--lr-scale', help='初期学習率のスケーリング係数を指定します。'
    'lr = default_lr * lr_scale / batch_size',
    type=int, default=0
)

parser.add_argument(
    '-np', '--num_progress', help='何層分周波数分解するかを指定します。(5~0)',
    type=int, default=5
)

parser.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='cifar10',
    choices=['mnist', 'fashion_mnist', 'cifar10', 'stl10', 'imagenet2012']
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default='~/.datasets/vision'
)

parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=999
)

# 入力に関するコマンドライン引数
parser.add_argument(
    '-id', '--input_dir', help='入力ディレクトリの名前を指定します。',
    type=str, default=None,
)

# 出力に関するコマンドライン引数
parser.add_argument(
    '--dir-name', help='出力ディレクトリの名前を指定します。',
    type=str, default=None,
)

parser.add_argument(
    '--nz', help='潜在空間の次元を指定します。',
    type=int, default=256
)

#   画像生成
parser.add_argument(
    '--num-samples', help='結果を見るための1クラス当たりのサンプル数を指定します。',
    type=int, default=49
)
parser.add_argument(
    '--sample-interval', help='生成画像の保存間隔をエポック数で指定します。',
    type=int, default=10,
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

parser.add_argument(
    '--save', help='訓練したモデルを保存します。',
    action='store_true'
)

parser.add_argument(
    '--lossy', help='可逆モードで実行します。',
    action='store_true'
)

parser.add_argument(
    '--lg', help='指定したパスのGeneratorのセーブファイルを読み込みます。',
    action='store_true'
)
parser.add_argument(
    '--ld', help='指定したパスのDiscriminatorのセーブファイルを読み込みます。',
    action='store_true'
)

parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)
# コマンドライン引数をパースする
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
LAUNCH_DATETIME = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# 名前を指定してロガーを取得する
logger = getLogger('main')
batch_size = args.batch_size
num_progress = args.num_progress
num_epochs = args.num_epochs
workers = 2
nc = 3
lr_scale = args.lr_scale
nz = args.nz
lr_g = 0.001
ngf = 64
lr_d = 0.001
ndf = 64

if args.input_dir is not None:
    INPUT_DIR = Path(
        f'./outputs/{args.dataset}/{args.input_dir}/models/{num_progress+1}')

if args.lg:
    load_generator = INPUT_DIR.joinpath('generator.pt')

if args.ld:
    load_discriminator = INPUT_DIR.joinpath('discriminator.pt')

# 出力に関する定数
if args.dir_name is None:
    OUTPUT_DIR = Path(
        LAUNCH_DATETIME.strftime(
            f'./outputs/{args.dataset}/%Y%m%d%H%M%S'))
else:
    OUTPUT_DIR = Path(f'./outputs/{args.dataset}/{args.dir_name}')
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
f_outputs = open(
    OUTPUT_DIR.joinpath('outputs.txt'), mode='w', encoding='utf-8')
f_outputs.write(' '.join(sys.argv) + '\n')
OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
OUTPUT_SAMPLE_DIR.mkdir(parents=True)
logger.info(f'画像用のディレクトリ({OUTPUT_SAMPLE_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

# 乱数生成器のシード値の設定
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# TODO: 完成したらコメントを外す
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
logger.info('乱数生成器のシード値を設定しました。')

device = 'cuda'
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')

logger.info('画像に適用する変換のリストを定義します。')
data_transforms = []
to_tensor = transforms.ToTensor()
data_transforms.append(to_tensor)

normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
data_transforms.append(normalize)
logger.info('変換リストに正規化を追加しました。')

# dataset = dset.STL10(
#            root=args.data_path, split='train',
#            transform=transforms.Compose(data_transforms), download=True)

dataset = dset.CIFAR10(
            root=args.data_path, train=True,
            transform=transforms.Compose(data_transforms), download=True)

# データセットの1番目の画像から色数を取得
nc, h, w = dataset[0][0].size()  # dataset[0][0].size() = (C, H, W)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    shuffle=True, drop_last=True, num_workers=workers)
logger.info('データローダを生成しました。')

print(len(dataset))


# =========================================================================== #
# モデルの定義
# =========================================================================== #
model_g = Generator(
    nz=nz, nc=nc
    ).to(device)
print(model_g)
# パラメータのロード
if args.lg:
    checkpoint = torch.load(load_generator)
    state_dict = checkpoint['model_state_dict']
    model_g.load_state_dict(state_dict)
    logger.info('Generatorのパラメータをロードしました。')

model_d = Discriminator(
    nc=nc
    ).to(device)
print(model_d)
# パラメータのロード
if args.ld:
    checkpoint = torch.load(load_discriminator)
    state_dict = checkpoint['model_state_dict']
    model_d.load_state_dict(state_dict)
    logger.info('Discriminatorのパラメータをロードしました。')


# =========================================================================== #
# オプティマイザの定義
# =========================================================================== #
optim_g = torch.optim.Adam(
    model_g.parameters(),
    lr=lr_g,
    betas=[0.5, 0.999]
    )

optim_d = torch.optim.Adam(
    model_d.parameters(),
    lr=lr_d,
    betas=[0.5, 0.999]
    )

sample_z = torch.randn(args.num_samples, nz, device=device)

f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
csv_writer = csv.writer(f_results, lineterminator='\n')
result_items = [
    'Epoch',
    'Generator Loss Mean', 'Discriminator Loss Mean',
    'Train Elapsed Time'
]
csv_writer.writerow(result_items)
csv_idx = {item: i for i, item in enumerate(result_items)}

if args.dct:
    wavelet = dct.DCT(num_progress)
elif args.fft:
    wavelet = fft.FFT()
else:
    wavelet = dwt.DWT(args.lossy)
wavelet = wavelet.to(device)

# =========================================================================== #
# 訓練
# =========================================================================== #
for epoch in range(num_epochs):
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    log_loss_g, log_loss_d = [], []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f'[{epoch+1}/{num_epochs}] 訓練開始',
        total=len(dataset)//batch_size,
        leave=False)
    model_g.train()  # Generatorを訓練モードに切り替える
    model_d.train()  # Discriminatorを訓練モードに切り替える
    begin_time = perf_counter()  # 時間計測開始
    for i, (real_images, _) in pbar:
        real_images = real_images.to(device)
        real_images = wavelet(real_images, num_progress)
        z = torch.randn(batch_size, nz, device=device)
        fake_images = model_g(z)

        #######################################################################
        # Discriminatorの訓練
        #######################################################################
        model_d.zero_grad()
        # Real画像についてDを訓練,
        pred_d_real = model_d(real_images)
        loss_d_real = F.relu(1.0 - pred_d_real).mean()

        # Fake画像についてDを訓練
        pred_d_fake = model_d(fake_images, detach=True)
        loss_d_fake = F.relu(1.0 + pred_d_fake).mean()

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        log_loss_d.append(loss_d.item())
        optim_d.step()

        #######################################################################
        # Generatorの訓練
        #######################################################################
        model_g.zero_grad()
        pred_g = model_d(fake_images)
        loss_g = -pred_g.mean()
        loss_g.backward()
        log_loss_g.append(loss_g.item())
        optim_g.step()

        # プログレスバーの情報を更新
        pbar.set_description_str(
            f'[{epoch+1}/{num_epochs}] 訓練中... '
            f'<損失: (G={loss_g.item():.016f}, D={loss_d.item():.016f})>')
    end_time = perf_counter()  # 時間計測終了
    pbar.close()

    loss_g_mean = np.mean(log_loss_g)
    loss_d_mean = np.mean(log_loss_d)
    results[csv_idx['Generator Loss Mean']] = f'{loss_g_mean:.016f}'
    results[csv_idx['Discriminator Loss Mean']] = f'{loss_d_mean:.016f}'

    train_elapsed_time = end_time - begin_time
    results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    print(
        f'[{epoch+1}/{num_epochs}] 訓練完了. '
        f'<エポック処理時間: {train_elapsed_time:.07f}[s/epoch]'
        f', 平均損失: (G={loss_g_mean:.016f}, D={loss_d_mean:.016f})>')

    model_g.eval()
    model_d.eval()

    if (
        epoch == 0
        or (epoch + 1) % args.sample_interval == 0
        or epoch == num_epochs - 1
    ):
        sample_dir = OUTPUT_SAMPLE_DIR.joinpath(f'{epoch + 1}')
        sample_dir.mkdir()
        with torch.no_grad():
            sample_images = model_g(sample_z).cpu()
            vutils.save_image(
                sample_images,
                sample_dir.joinpath(f'{epoch}.png'),
                nrow=int(np.sqrt(args.num_samples)),
                range=(-1.0, 1.0),
                normalize=True
                )
            logger.info('画像を生成しました。')
    csv_writer.writerow(results)
    f_results.flush()
OUTPUT_MODEL_DIR = OUTPUT_MODEL_DIR.joinpath(f'{num_progress}')
if args.save and (epoch == num_epochs - 1):

    OUTPUT_MODEL_DIR.mkdir(exist_ok=True)  # モデルの出力ディレクトリを作成
    torch.save(  # Generatorのセーブ
        {
            'model_state_dict': model_g.state_dict(),
            'optimizer_state_dict': optim_g.state_dict(),
            'lrs_state_dict': lr_g,
            'last_epoch': epoch,
            'batch_size': batch_size,
            'dataset': args.dataset,
            'nz': nz,
            'nc': nc,
            'lossy': args.lossy
        },
        OUTPUT_MODEL_DIR.joinpath('generator.pt')
    )
    torch.save(  # Discriminatorのセーブ
        {
            'model_state_dict': model_d.state_dict(),
            'optimizer_state_dict': optim_d.state_dict(),
            'lrs_state_dict': lr_d,
            'last_epoch': epoch,
            'batch_size': batch_size,
            'dataset': args.dataset,
            'nc': nc,
            'lossy': args.lossy
        },
        OUTPUT_MODEL_DIR.joinpath('discriminator.pt')
    )
f_results.close()
f_outputs.close()
