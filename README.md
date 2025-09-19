# Adversarial Robustness Evaluation for Time-Series Foundation Models

This repository contains the official implementation of the paper:  
**"Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains"**


---

## 🔧 Installation

We recommend using a dedicated Python environment:

```bash
conda create --name attack_fm python=3.11
conda activate attack_fm
pip install .
pip install autogluon
pip install tabpfn_time_series
```

## 📦 Download Model Checkpoints

To evaluate pre-trained models, download the required checkpoints using the following script:

```bash
bash download_ckpt.sh
```

## 📂 Download Datasets from GIFT-EVAL Benchmark

This evaluation uses datasets from the [GIFT-EVAL Benchmarking](https://github.com/SalesforceAIResearch/gift-eval?tab=readme-ov-file#installation).

Follow the steps below to download and install the necessary files:

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval.git
cd gift-eval
pip install -e .
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir PATH_TO_DATA
```

Note: The dataset does not need to be stored in this project folder. You only need to install the gift-eval package and download the datasets to a local directory of your choice (`PATH_TO_DATA`). Be sure to update the dataset path in your configuration accordingly when running experiments.

## 🚀 Quick Start

To quickly test our framework and see results, run the [demo notebook](./demo.ipynb):


The notebook demonstrates how to:
- Load a pre-trained TSFM
- Apply adversarial attacks (e.g., PGD)
- Visualize the impact on predictions


## 🛡 Defense Methods

We include two defense families:
1. Inference-Time Smoothing (no retraining)
2. Latent Adversarial Training (LAT) for TimesFm

Both are evaluated against clean, PGD, and SimBA settings and report ND / NRMSE / CRPS.

**1) Inference-Time Smoothing (no retraining)**

```bash
python ./defense/inference_smooth_timesfm.py \
  --dataset ett1/H/short \
  --dataset_path PATH_TO_DATA \
  --context_length 96 \
  --batch_size 64 \
  --model_size 200m \
  --kernels 1,3,5,7 \
  --robust_eval true \
  --attack_gt false \
  --attack_ratio 1.0 \
  --epsilon 0.5 \
  --alpha 0.05 \
  --num_iterations 50 \
  --simba_mode random \
  --simba_basis point \
  --norm linf \
  --gift_eval true \
  --save_dir ./smooth_eval_logs 
```

**2) Latent Adversarial Training (LAT)**

We perturb the latent activations during training and minimize the worst-case loss w.r.t bounded latent perturbations. We fine-tune TimesFm.

```bash
python ./defense/finetune_lat_timesfm.py \
  --dataset ett1/H/short \
  --dataset_path PATH_TO_DATA \
  --context_length 96 \
  --batch_size 64 \
  --model_size 200m \
  --epochs 15 \
  --lr 1e-4 \
  --inner_steps 10 \
  --eta_delta 0.5 \
  --eps 0.5 \
  --p_norm linf \
  --gift_eval true \
  --save_dir ./smooth_eval_logs \
  --max_iter 10
```
