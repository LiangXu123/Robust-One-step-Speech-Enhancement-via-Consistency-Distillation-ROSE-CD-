from scipy.io import wavfile

# Load WAV file
file_ = '/home/liangxu/data/voicebank/train/clean/p243_035.wav'

sampling_rate, _ = wavfile.read(file_)

# Print sampling rate in kHz
print(f"Sampling Rate: {sampling_rate / 1000:.1f} kHz")


# Load WAV file
output_file = '/home/liangxu/data/voicebank/train/noisy/p243_035.wav'
sampling_rate, _ = wavfile.read(output_file)

# Print sampling rate in kHz
print(f"Sampling Rate: {sampling_rate / 1000:.1f} kHz")
