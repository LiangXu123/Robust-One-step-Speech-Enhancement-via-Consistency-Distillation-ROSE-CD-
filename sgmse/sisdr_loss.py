from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses import PITLossWrapper
import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
import numpy as np
import torch
from torchmetrics import ScaleInvariantSignalDistortionRatio

# Load wav
noisy_file = '/home/liangxu/data/voicebank/test/noisy/p232_001.wav'
y, sr = load(noisy_file)
# Load audio data from source.
# Normalize
norm_factor = y.abs().max()
y = y / norm_factor


target = torch.tensor([3.0, -0.5, 2.0, 7.0])
preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
si_sdr = ScaleInvariantSignalDistortionRatio()
si_sdr(preds, target)
si_sdr(y, y*2)

b = np.ones((8, y.shape[1])) + 1.2
a = np.ones((8, y.shape[1]))*0.648
a = torch.from_numpy(a)
b = torch.from_numpy(b)
si_sdr(a, b)


targets = torch.randn(10, 2, 32000)
est_targets = torch.randn(10, 2, 32000)
loss_func = PITLossWrapper(pairwise_neg_sisdr,
                           pit_from='pw_mtx')
loss = loss_func(est_targets, targets)

loss_func(est_targets, targets)
loss_func(est_targets*2, targets)
loss_func(targets, targets)
loss_func(targets, targets*2)
