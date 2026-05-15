#!/bin/bash

# =======================
# Default path candidates
# =======================

# --- VoiceBank test set ---
export CLEAN_DIR_VOICEBANK="/home/liangxu/data/voicebank/true_split_train_val_test/16K//test/clean"
export NOISY_DIR_VOICEBANK="/home/liangxu/data/voicebank/true_split_train_val_test/16K//test/noisy"
export BASE_DIR_VOICEBANK="/home/liangxu/data/voicebank/true_split_train_val_test/16K/"

# --- TIMIT + NOISEX92 test set ---
export CLEAN_DIR_TIMIT="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/complete_test/clean"
export NOISY_DIR_TIMIT="/vol/grid-solar/sgeusers/liangxu/data/TIMIT_NOISEX92/complete_test/noisy"

# --- DNS-Challenge test set ---
export INPUT_DIR_DNS="/vol/grid-solar/sgeusers/liangxu/exp_code/Eval_waspaa2025/DNS_Challenge_test_set_real_recordings_16khz"

# --- DNS-Challenge2022 test set ---
export INPUT_DIR_DNS22="/vol/grid-solar/sgeusers/liangxu/data/DNS-Challenge/V5_BlindTestSet_mono/test_all16khz/"
# toolong_6mins
# test_all16khz

# --- SIG Challenge 2024 test set ---
export INPUT_DIR_SIG24="/vol/grid-solar/sgeusers/liangxu/data/SIG-Challenge/ICASSP2024/blind_testset16khz/"
