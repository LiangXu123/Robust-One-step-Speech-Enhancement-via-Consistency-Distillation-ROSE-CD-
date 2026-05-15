import librosa
import soundfile as sf


def resample_wav_librosa(input_path, output_path, target_sr=16000):
    # Load audio file and resample
    audio, sr = librosa.load(input_path, sr=target_sr)

    # Save the resampled audio
    sf.write(output_path, audio, target_sr)

    print(
        f"Resampled (Librosa): {input_path} → {output_path} at {target_sr} Hz")


# Load WAV file
input_file = '/home/liangxu/data/voicebank/train/clean/p287_371.wav'
output_file = '/home/liangxu/data/voicebank/train/p287_371.wav'

# Example usage
resample_wav_librosa(input_file, output_file)
