# ROSE-CD

This repository provides the official PyTorch implementation of the following paper:

**Robust One-step Speech Enhancement via Consistency Distillation (IEEE WASPAA 2025, Oral Presentation)**  
Liang Xu, Longfei Felix Yan, W. Bastiaan Kleijn  
*IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2025*

🔗 [**Project Website**](https://liangxu123.github.io/rosecd/)  |  📄 [**arXiv Preprint**](https://arxiv.org/abs/2507.05688)  |  📄 [**IEEE Xplore**](https://ieeexplore.ieee.org/document/11230988)

---

## 📖 Highlights

- **Real-Time Efficiency:** Proposes a one-step consistency training (CT) framework for highly efficient, real-time speech enhancement.
- **Improved Robustness:** Mitigates the accumulation of teacher-induced biases via randomized trajectory training and auxiliary time-domain constraints.
- **Superior Performance:** Accelerates inference speed by a factor of 54× while surpassing the performance of the foundational 30-step teacher model.
- **Strong Generalization:** Demonstrates robust generalization capabilities across out-of-domain and dynamic real-world acoustic scenarios.

---

## 📊 Performance Benchmark

The table below presents a comparative analysis between the proposed **1-step Consistency Training (CT)** model and the baseline **30-step Teacher** model, evaluated on the VoiceBank-DEMAND test corpus. The CT framework not only accelerates inference by an order of magnitude but also yields statistically significant improvements across all established objective metrics.

| Model | Steps | PESQ (↑) | ESTOI (↑) | SI-SDR (↑) | SI-SIR (↑) | SI-SAR (↑) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher** | 30 | 2.89 ± 0.67 | 0.86 ± 0.10 | 16.7 ± 3.7 | 26.7 ± 5.8 | 17.6 ± 3.4 |
| **CT (Ours)** | **1** | **3.47** ± 0.67 | **0.87** ± 0.10 | **19.2** ± 3.6 | **29.2** ± 5.4 | **20.0** ± 3.7 |

---

## 🔗 Pre-trained Models & Audio Samples

We release the pre-trained checkpoints alongside corresponding enhanced audio outputs for both the 30-step teacher model and our 1-step CT model.

- **Checkpoints**: [Download via Google Drive](https://drive.google.com/file/d/1ekzJQidIojhjlj6oaUzQBKp4Pil6jIz7/view?usp=sharing)
- **Enhanced Audio Outputs**: [Download via Google Drive](https://drive.google.com/file/d/17hyzn2CWzzpDg44spLp8NJJbiDK4v_wN/view?usp=sharing)

**Usage instructions:**
Extract and place the downloaded checkpoints into the designated `logs/` directory (e.g., `./logs/`). Ensure that the checkpoint paths within the evaluation scripts (`scripts/eval_CT.sh` or `scripts/eval_teacher.sh`) are correctly updated to replicate the reported benchmark results.

---

## ⚙️ Installation & Setup

We recommend utilizing an isolated virtual environment with Python 3.11. To initialize the environment and install dependencies, execute:

```bash
# Clone the repository
git clone https://github.com/liangxu123/rosecd.git
cd rosecd

# Install required packages
pip install -r requirements.txt
```

*Note: For experiment tracking via Weights & Biases (W&B), please configure your environment using `wandb login` prior to initiating training.*

---

## 🗄️ Dataset Preparation

Our data preprocessing pipeline is adapted from the established [SGMSE+](https://github.com/sp-uhh/sgmse) framework. To configure the dataset directories, please update the corresponding paths in `path_config.sh`. By default, the configuration points to the VoiceBank-DEMAND corpus paths.

---

## 🚀 Training Framework: Consistency Training (CT) and Consistency Distillation (CD)

Consistency Models inherently support two distinct training paradigms: 
1. **Consistency Distillation (CD):** Distilling knowledge from a pre-trained teacher diffusion model.
2. **Consistency Training (CT):** Direct training on the empirical data distribution without the necessity of a teacher model.

While our published paper primarily formalizes and evaluates the method utilizing **Consistency Distillation (CD)**, subsequent empirical analyses revealed that applying **Consistency Training (CT)** within the exact same codebase yields identical performance. Because CT completely bypasses the need to rely on a pre-trained teacher model, it significantly streamlines the training procedure and circumvents teacher-induced approximation errors. 

Therefore, in this repository, we officially release both the **Teacher model training** scripts and the **CT model training** scripts, as CT achieves the same state-of-the-art one-step enhancement performance as CD, but with a much simpler pipeline.

To train the one-step model from scratch using CT, execute:

```bash
bash ./scripts/train_CT.sh <GPU_ID>  # e.g., bash ./scripts/train_CT.sh 0
```

To train the multi-step Teacher model (if you wish to reproduce the baseline or teacher pipeline), execute:

```bash
bash ./scripts/train_teacher.sh <GPU_ID>  # e.g., bash ./scripts/train_teacher.sh 0
```

---

## 📈 Evaluation Protocol

To benchmark the one-step consistency model on the test corpus, utilize the evaluation scripts provided in the `scripts/` directory.

Prior to execution, verify that the checkpoint path inside `scripts/eval_CT.sh` correctly points to your trained model weights.

```bash
bash ./scripts/eval_CT.sh <GPU_ID>  # e.g., bash ./scripts/eval_CT.sh 0
```

This procedure generates the enhanced audio files within `out/onestep_pesq5e-4_CT/N_1` and subsequently computes standard objective metrics (e.g., PESQ, ESTOI, SI-SDR).

**Evaluating the Baseline Teacher Model:**  
If a baseline multi-step Teacher model was trained (via `scripts/train_teacher.sh`), it can be evaluated by updating the checkpoint path within `scripts/eval_teacher.sh` and running:

```bash
bash ./scripts/eval_teacher.sh <GPU_ID>  # e.g., bash ./scripts/eval_teacher.sh 0
```

Enhanced outputs will be saved to `out/teacher/N_30` and evaluated automatically.

---

## 📝 Citation

If this codebase or methodology proves useful in your research, please consider citing our work:

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

## 🙏 Acknowledgments

We express our gratitude to the authors of the [SGMSE+](https://github.com/sp-uhh/sgmse) repository, upon whose exemplary foundational work this codebase is built.
