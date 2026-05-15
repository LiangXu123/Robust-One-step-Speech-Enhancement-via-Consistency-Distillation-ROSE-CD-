from warpq.utils import load_audio, plot_warpq_scores, group_dataframe_by_columns
from warpq.core import warpqMetric
from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std

import os

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
config = visqol_config_pb2.VisqolConfig()
# Create an instance of the warpqMetric class
model_warp = warpqMetric()

mode = "speech"
if mode == "audio":
    config.audio.sample_rate = 48000
    config.options.use_speech_scoring = False
    svr_model_path = "libsvm_nu_svr_model.txt"
elif mode == "speech":
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "/workspace/exp_code/visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
else:
    raise ValueError(f"Unrecognized mode: {mode}")

config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), svr_model_path)

visqol_api = visqol_lib_py.VisqolApi()

visqol_api.Create(config)

# reference, sr_x = read('/workspace/data/voicebank/test/clean/p232_001.wav')
# degraded, sr_x = read(
#     '/workspace/exp_code/CausalRose-CD/out/Onestep_pesq5e-4realtime_20_5/N_1/p232_001.wav')
# degraded = librosa.resample(
#     degraded, orig_sr=sr_x, target_sr=16000)
# reference = librosa.resample(
#     reference, orig_sr=sr_x, target_sr=16000)

# moslqo_score = visqol_api.Measure(reference, degraded).moslqo
# print(moslqo_score)


# Evaluate the audio quality between two files
# results = model_warp.evaluate('/workspace/data/voicebank/test/clean/p232_001.wav',
#                               '/workspace/exp_code/CausalRose-CD/out/Onestep_pesq5e-4realtime_20_5/N_1/p232_001.wav', verbose=True)

# # Access the raw WARP-Q and normalized scores
# raw_warpq_score = results["raw_warpq_score"]
# normalized_warpq_score = results["normalized_warpq_score"]

# Print the results
# print(f"Raw WARP-Q Score: {raw_warpq_score}")
# print(f"Normalized WARP-Q Score: {normalized_warpq_score}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                        help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help='Directory containing the enhanced data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "moslqo": [], "normalized_warpq_score": [], "estoi": [],
            "si_sdr": [], "si_sir": [],  "si_sar": []}

    # Evaluate standard metrics
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        x, sr_x = read(join(args.clean_dir, clean_filename))
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))
        assert sr_x == sr_y == sr_x_hat
        n = y - x
        x_hat_16k = librosa.resample(
            x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(
            x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["moslqo"].append(visqol_api.Measure(x_16k, x_hat_16k).moslqo)
        data["normalized_warpq_score"].append(
            model_warp.evaluate(x_16k, x_hat_16k, arr_sr=16000, verbose=False)["normalized_warpq_score"])
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("MOSLQO: {:.2f} ± {:.2f}".format(*mean_std(df["moslqo"].to_numpy())))
    print("WARP-Q: {:.2f} ± {:.2f}".format(
        *mean_std(df["normalized_warpq_score"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(
        *mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("MOSLQO(ViSQOL): {:.2f} ± {:.2f}".format(
        *mean_std(df["moslqo"].to_numpy())) + "\n")
    log.write("WARP-Q: {:.2f} ± {:.2f}".format(
        *mean_std(df["normalized_warpq_score"].to_numpy())) + "\n")
    log.write("ESTOI: {:.2f} ± {:.2f}".format(
        *mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SI-SDR: {:.1f} ± {:.2f}".format(*
              mean_std(df["si_sdr"].to_numpy())) + "\n")
    log.write("SI-SIR: {:.1f} ± {:.1f}".format(*
              mean_std(df["si_sir"].to_numpy())) + "\n")
    log.write("SI-SAR: {:.1f} ± {:.1f}".format(*
              mean_std(df["si_sar"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)
