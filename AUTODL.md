# AutoDL execution guide

The verified instance environment is RTX 4090 24 GB, Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, and CUDA 11.8.

## Install

```bash
cd /root/autodl-tmp/Grasp-Anything-ProgrammingTest
pip install -r requirements-autodl.txt
export HF_ENDPOINT=https://hf-mirror.com
```

## Prepare a compact official subset

Download only the two small Grasp-Anything++ archives used by the method:

```bash
mkdir -p /root/autodl-tmp/grasp_anything_pp/archives
wget -c -O /root/autodl-tmp/grasp_anything_pp/archives/grasp_instructions.zip \
  https://hf-mirror.com/datasets/airvlab/Grasp-Anything-pp/resolve/main/grasp_instructions.zip
wget -c -O /root/autodl-tmp/grasp_anything_pp/archives/grasp_label_positive.zip \
  https://hf-mirror.com/datasets/airvlab/Grasp-Anything-pp/resolve/main/grasp_label_positive.zip
```

Prepare 1,000 scenes with up to three language instructions per scene. Images are range-read from the official split ZIP, so the 65 GB image archive is never stored locally.

```bash
python tools/prepare_ga_pp_subset.py \
  --instructions-zip /root/autodl-tmp/grasp_anything_pp/archives/grasp_instructions.zip \
  --labels-zip /root/autodl-tmp/grasp_anything_pp/archives/grasp_label_positive.zip \
  --output-root /root/autodl-tmp/grasp_anything_pp/subset_1k \
  --max-scenes 1000 \
  --samples-per-scene 3
```

The split is deterministic and scene-disjoint. A negative prompt targets another object or part in the same scene whenever possible.

## Train

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train_full.py \
  --train-manifest /root/autodl-tmp/grasp_anything_pp/subset_1k/train.jsonl \
  --val-manifest /root/autodl-tmp/grasp_anything_pp/subset_1k/val.jsonl \
  --output-dir /root/autodl-tmp/pcghnet_runs/baseline \
  --batch-size 32 \
  --epochs 20
```

For a short pipeline check, add `--text-encoder hash --no-pretrained-backbone --max-steps 5 --epochs 1`.

## Evaluate

```bash
python evaluate.py \
  --checkpoint /root/autodl-tmp/pcghnet_runs/baseline/best.pt \
  --manifest /root/autodl-tmp/grasp_anything_pp/subset_1k/val.jsonl \
  --batch-size 32 \
  --output /root/autodl-tmp/pcghnet_runs/baseline/metrics.json
```

The evaluator reports rectangle success rate (IoU at least 0.25 and angle error at most 30 degrees), mean best IoU, mean angle error, and the correct-vs-negative target prompt gap.
