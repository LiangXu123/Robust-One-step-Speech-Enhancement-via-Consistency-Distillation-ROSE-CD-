#!/bin/bash

# =======================
# Default path candidates
# =======================

# --- VoiceBank test set ---
export CLEAN_DIR_VOICEBANK="/vol/grid-solar/sgeusers/liangxu/data/voicebank/mixed_snr0/test/clean/"
export NOISY_DIR_VOICEBANK="/vol/grid-solar/sgeusers/liangxu/data/voicebank/mixed_snr0/test/noisy/"
export BASE_DIR_VOICEBANK="/vol/grid-solar/sgeusers/liangxu/data/voicebank/mixed_snr0"

# --- TIMIT + NOISEX92 test set ---
export CLEAN_DIR_TIMIT="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/complete_test/clean"
export NOISY_DIR_TIMIT="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/complete_test/noisy"
export NOISY_DIR_TIMIT_REGULAR="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/comparative_stress_test/test/noisy_regular/"
export NOISY_DIR_TIMIT_HIGHFREQ="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/comparative_stress_test/test/noisy_hf_stress/"

# --- DNS-Challenge test set ---
export INPUT_DIR_DNS="/vol/grid-solar/sgeusers/liangxu/data/DNS_Challenge_test_set_real_recordings_16khz"
export INPUT_DIR_DNS_hardcut6khz="/vol/grid-solar/sgeusers/liangxu/data/DNS_Challenge_test_set_real_recordings_16khz_hardcut6khz"