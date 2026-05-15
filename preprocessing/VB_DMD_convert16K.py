import os
from pydub import AudioSegment


def resample_folder_pydub(input_folder, output_folder, target_sr=16000):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load and resample
            audio = AudioSegment.from_wav(input_path)
            audio = audio.set_frame_rate(target_sr)
            audio.export(output_path, format="wav")

            print(f"Resampled (pydub): {filename} → {target_sr} Hz")


# Example usage
# resample_folder_pydub('/home/liangxu/data/voicebank/test/noisy/',
            #   "/home/liangxu/data/voicebank/test/noisy16khz")
resample_folder_pydub('/local/scratch/TIMIT/complete_testset/',
                      "/local/scratch/TIMIT/complete_testset16khz")
