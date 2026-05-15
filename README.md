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

## Performance Results

The following table demonstrates the performance leap of the **1-step Consistency Training (CT)** model compared to the **30-step Teacher** model on the VoiceBank-DEMAND test set. The CT model not only accelerates inference by an order of magnitude but also significantly improves all objective metrics.

| Model | Steps | PESQ (↑) | ESTOI (↑) | SI-SDR (↑) | SI-SIR (↑) | SI-SAR (↑) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher** | 30 | 2.89 ± 0.67 | 0.86 ± 0.10 | 16.7 ± 3.7 | 26.7 ± 5.8 | 17.6 ± 3.4 |
| **CT (Ours)** | **1** | **3.47** ± 0.67 | **0.87** ± 0.10 | **19.2** ± 3.6 | **29.2** ± 5.4 | **20.0** ± 3.7 |

---

## Pre-trained Models & Enhanced Outputs

We provide the pre-trained checkpoints and the corresponding enhanced audio outputs for both the 30-step Teacher model and our 1-step CT model.

- **Checkpoints**: [Google Drive](https://drive.google.com/file/d/1ekzJQidIojhjlj6oaUzQBKp4Pil6jIz7/view?usp=sharing)
- **Enhanced Outputs**: [Google Drive](https://drive.google.com/file/d/17hyzn2CWzzpDg44spLp8NJJbiDK4v_wN/view?usp=sharing)

After downloading the checkpoints, place them in the appropriate directory (e.g., `./logs/`) and update the checkpoint paths in the evaluation scripts (`eval_CT.sh` or `eval_teacher.sh`) to reproduce our results.

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

The data processing follows the exact same method as the [SGMSE+](https://github.com/sp-uhh/sgmse) repository. You just need to update the dataset paths for training and testing in `path_config.sh`.
By default, the scripts point to paths for VoiceBank-DEMAND.

---

## Training Pipeline: Consistency Training (CT)

We provide a direct, single-step training pipeline called **Consistency Training (CT)**.

> **Why CT instead of CD?**  
> While Consistency Distillation (CD) requires training a heavy "Teacher" diffusion model and distilling its knowledge, **Consistency Training (CT)** directly enforces self-consistency on the true data distribution. By mapping any noisy point on the Probability Flow ODE trajectory straight to its clean origin, CT entirely bypasses the need for a teacher network. This eliminates teacher-induced approximation errors, drastically simplifies the training pipeline, and delivers identical (or better) state-of-the-art one-step enhancement performance.

To train the one-step model directly from scratch:

```bash
bash ./scripts/train_CT.sh 0 # where 0 is the GPU_ID
```

---

## Evaluation

To evaluate the one-step consistency model on test sets, use the provided scripts in the `scripts/` directory.

Before running the evaluation, remember to update the checkpoint path inside `scripts/eval_CT.sh` to point to your trained model.

```bash
bash ./scripts/eval_CT.sh 0 # where 0 is the GPU_ID
```

This script will automatically generate the enhanced audio in `out/onestep_pesq5e-4_CT/N_1` and immediately evaluate it using the objective metrics scripts (calculating PESQ, ESTOI, SI-SDR, etc.).

**Evaluating the Baseline Teacher Model**  
If you trained a baseline multi-step Teacher model (using `train_teacher.sh`), you can evaluate it by first setting your checkpoint path inside `scripts/eval_teacher.sh` and then running:

```bash
bash ./scripts/eval_teacher.sh 0 # where 0 is the GPU_ID
```

This script will automatically generate the enhanced audio in `out/teacher/N_30` and immediately evaluate it using the standard metric calculation script.

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

---

## Acknowledgments

We would like to thank the authors of the [SGMSE+](https://github.com/sp-uhh/sgmse) repository. Our codebase is built upon their excellent work.
