import os
from glob import glob
from librosa import load
from librosa.core import resample
import argparse
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from soundfile import write
from tqdm import tqdm
import random

# Python script for generating noisy mixtures for training
#
# Mix WSJ0 with CHiME3 noise with SNR sampled uniformly in [min_snr, max_snr]


SNR_list = [0, 5, 10, 15]
sr = 16000


if __name__ == '__main__':

    # Define default paths (adjust these to your actual directories)
    # Your 192 clean clips
    DEFAULT_TIMIT_DIR = '/home/liangxu/data/TIMIT_NOISEX92/complete_testset16khz'
    # Your 15 noise samples
    DEFAULT_NOISE_DIR = '/home/liangxu/data/TIMIT_NOISEX92/NoiseX-92'
    # Where the test set will go
    DEFAULT_TARGET_DIR = '/home/liangxu/data/TIMIT_NOISEX92/complete_test'

    # Set up argument parser with defaults
    parser = argparse.ArgumentParser(
        description="Create a TIMIT+NOISE92 test set")
    parser.add_argument(
        "TIMIT",
        nargs="?",  # Makes it optional
        default=DEFAULT_TIMIT_DIR,
        help=f"Path to TIMIT audio directory (default: {DEFAULT_TIMIT_DIR})"
    )
    parser.add_argument(
        "noise",
        nargs="?",
        default=DEFAULT_NOISE_DIR,
        help=f"Path to noise samples directory (default: {DEFAULT_NOISE_DIR})"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_TARGET_DIR,
        help=f"Path to output test set directory (default: {DEFAULT_TARGET_DIR})"
    )
    args = parser.parse_args()

    # Clean speech for training
    test_speech_files = sorted(
        glob(args.TIMIT + '/*.wav', recursive=True))

    noise_files = glob(args.noise + '/*.wav', recursive=True)

    # Load noise files
    noises = []
    print('Loading noise files')
    for file in noise_files:
        noise = load(file, sr=None)[0]
        noises.append(noise)

    # Create target dir
    test_clean_path = Path(os.path.join(args.target, 'test/clean'))
    test_noisy_path = Path(os.path.join(args.target, 'test/noisy'))

    test_clean_path.mkdir(parents=True, exist_ok=True)
    test_noisy_path.mkdir(parents=True, exist_ok=True)

    # Create files for test
    print('Create test files')
    for i, speech_file in enumerate(tqdm(test_speech_files)):
        s, _ = load(speech_file, sr=sr)

        snr_dB = random.choice(SNR_list)
        noise_ind = np.random.randint(len(noises))
        speech_power = 1/len(s)*np.sum(s**2)

        n = noises[noise_ind]
        start = np.random.randint(len(n)-len(s))
        n = n[start:start+len(s)]

        noise_power = 1/len(n)*np.sum(n**2)
        noise_power_target = speech_power*np.power(10, -snr_dB/10)
        k = noise_power_target / noise_power
        n = n * np.sqrt(k)
        x = s + n

        # file_name = speech_file.split('/')[-1]
        speech_file = str(speech_file).replace('WAV', 'wav')
        speech_file = str(speech_file).replace('_output', '')
        file_name = os.path.basename(speech_file)
        file_name = str(file_name).replace('converted_', '')
        write(os.path.join(test_clean_path, file_name), s, sr)
        write(os.path.join(test_noisy_path, file_name), x, sr)

        # for snr_dB in SNR_list:
        #     noise_id = 0
        #     for current_noise in noises:
        #         noise_id += 1
        #         # snr_dB = np.random.uniform(min_snr, max_snr)
        #         # noise_ind = np.random.randint(len(noises))
        #         speech_power = 1/len(s)*np.sum(s**2)

        #         n = current_noise
        #         start = np.random.randint(len(n)-len(s))
        #         n = n[start:start+len(s)]

        #         noise_power = 1/len(n)*np.sum(n**2)
        #         noise_power_target = speech_power*np.power(10, -snr_dB/10)
        #         k = noise_power_target / noise_power
        #         n = n * np.sqrt(k)
        #         x = s + n

        #         speech_file = str(speech_file).replace('WAV', 'wav')
        #         # file_name = speech_file.split('/')[-1]
        #         file_name = os.path.basename(speech_file)
        #         file_name = 'snr_' + str(snr_dB) + \
        #             '_noiseid_'+str(noise_id) + file_name
        #         write(os.path.join(test_clean_path, file_name), s, sr)
        #         write(os.path.join(test_noisy_path, file_name), x, sr)
