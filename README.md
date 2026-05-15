# Robust One-step Speech Enhancement via Consistency Distillation (ROSE-CD)(IEEE WASPAA ORAL)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/robust-one-step-speech-enhancement-via-1/speech-enhancement-on-demand)](https://paperswithcode.com/sota/speech-enhancement-on-demand?p=robust-one-step-speech-enhancement-via-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/robust-one-step-speech-enhancement-via-1/speech-enhancement-on-voicebank-demand-2)](https://paperswithcode.com/sota/speech-enhancement-on-voicebank-demand-2?p=robust-one-step-speech-enhancement-via-1)

This repository accompanies the following paper:

**Robust One-step Speech Enhancement via Consistency Distillation(IEEE WASPAA ORAL)**  
Liang Xu, Longfei Felix Yan, W. Bastiaan Kleijn  
*IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2025*

🔗 [**Project Website**](https://liangxu123.github.io/rosecd/)  
📄 [**Read the Paper (arXiv)**](https://arxiv.org/abs/2507.05688)
📄 [**Read the Paper (IEEE)**](https://ieeexplore.ieee.org/document/11230988)

---

## Highlights

- A one-step consistency distillation method for real-time speech enhancement.
- Mitigates teacher bias via randomized training and time-domain auxiliary losses.
- Achieves 54× faster inference while surpassing the 30-step teacher model.
- Demonstrates strong generalization across out-of-domain and real-world scenarios.

---

## Installation

Create a new virtual environment and install the required dependencies (Python 3.11 is recommended).

```bash
# Clone the repository
git clone https://github.com/liangxu123/rosecd.git
cd rosecd

# Install dependencies
pip install -r requirements.txt
```

If you are using W&B for logging, set up an account and run `wandb login` before training.

---

## Dataset Preparation

Set the correct dataset paths for training and testing in `path_config.sh`.
By default, the scripts point to paths for VoiceBank-DEMAND, TIMIT+NOISEX92, DNS-Challenge, and SIG Challenge.

---

## Teacher Training

Training the teacher model is done by executing `train.py`. A minimal running example with default settings (as in our paper but supporting the EDM architecture) can be run with:

```bash
bash ./scripts-Teacher/train_teacher_model.sh 0 # where 0 is the GPU_ID
```

---

## One-step Consistency Model Training

We provide scripts and results for both **Consistency Training (CT)** and **Consistency Distillation (CD)**. Both methods deliver excellent performance! 

> **Recommendation:** We highly encourage using **CT (Consistency Training)**. CT achieves the same high performance as CD, but it **does not require a pre-trained teacher model**, making the training pipeline much simpler!

Training the consistency model is done by executing `onestep_train.py`. A minimal running example (see `scripts-Onestep/onestep_train_pesq1e-3.sh` and `onestep_train_pesq5e-4.sh`) can be run with:

```bash
bash ./scripts-Onestep/onestep_train_pesq1e-3.sh 0 # where 0 is the GPU_ID
```

---

## Evaluation

To evaluate the one-step consistency model on test sets, use the provided scripts in the `scripts-Onestep/` directory.

For example, to evaluate on the VoiceBank-DEMAND test set:

```bash
bash ./scripts-Onestep/eval_onestep_VB.sh 0 # where 0 is the GPU_ID
```

For real-time evaluation:

```bash
bash ./scripts-Onestep/eval_onestep_VBrealtime.sh 0
```

---

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{xu2025robust,
  title={Robust One-step Speech Enhancement via Consistency Distillation},
  author={Xu, Liang and Yan, Longfei Felix and Kleijn, W Bastiaan},
  booktitle={2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages={1--5},
  year={2025},
  organization={IEEE},
  keywords={Noise;Speech enhancement;Robustness;Real-time systems;Trajectory;Recording;Noise measurement;Iterative methods;Time-domain analysis;Optimization},
  doi={10.1109/WASPAA66052.2025.11230988}
}
```
