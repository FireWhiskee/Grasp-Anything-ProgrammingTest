# Reproduce the final experiments

All runs use the same scene-disjoint split, ImageNet-pretrained ResNet-18 backbone, seed 7, optimizer, and 20-epoch schedule. Only language conditioning and the proposed consistency loss change.

## 1. Download annotations

```bash
mkdir -p /root/autodl-tmp/grasp_anything_pp/archives
wget -c -O /root/autodl-tmp/grasp_anything_pp/archives/grasp_instructions.zip \
  https://hf-mirror.com/datasets/airvlab/Grasp-Anything-pp/resolve/main/grasp_instructions.zip
wget -c -O /root/autodl-tmp/grasp_anything_pp/archives/grasp_label_positive.zip \
  https://hf-mirror.com/datasets/airvlab/Grasp-Anything-pp/resolve/main/grasp_label_positive.zip
unzip -t /root/autodl-tmp/grasp_anything_pp/archives/grasp_instructions.zip
unzip -t /root/autodl-tmp/grasp_anything_pp/archives/grasp_label_positive.zip
```

The preparation script range-reads only selected JPEGs from the official image archive; it does not store the full 65 GB archive.

## 2. Prepare 5,000 scenes

```bash
cd /root/autodl-tmp/Grasp-Anything-ProgrammingTest
python tools/prepare_ga_pp_subset.py \
  --instructions-zip /root/autodl-tmp/grasp_anything_pp/archives/grasp_instructions.zip \
  --labels-zip /root/autodl-tmp/grasp_anything_pp/archives/grasp_label_positive.zip \
  --output-root /root/autodl-tmp/grasp_anything_pp/subset_5k \
  --max-scenes 5000 \
  --samples-per-scene 3 \
  --validation-ratio 0.1 \
  --test-ratio 0.1
```

The resulting scene-disjoint manifests contain 11,588 training, 1,413 validation, and 1,458 test samples.

## 3. Train the three ablations

```bash
COMMON="--train-manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/train.jsonl \
--val-manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/val.jsonl \
--batch-size 32 --epochs 20 --seed 7"

python train_full.py $COMMON \
  --output-dir /root/autodl-tmp/pcghnet_runs/image_only \
  --image-only --no-negative-prompts

python train_full.py $COMMON \
  --output-dir /root/autodl-tmp/pcghnet_runs/film \
  --no-negative-prompts

python train_full.py $COMMON \
  --output-dir /root/autodl-tmp/pcghnet_runs/pcghnet_full_v2 \
  --consistency-weight 0.1
```

Keep `best.pt`, selected only by validation success rate.

## 4. Evaluate the held-out test split

```bash
for RUN in image_only film pcghnet_full_v2; do
  python evaluate.py \
    --checkpoint /root/autodl-tmp/pcghnet_runs/$RUN/best.pt \
    --manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/test.jsonl \
    --batch-size 32 \
    --output /root/autodl-tmp/pcghnet_runs/$RUN/test_metrics.json
done
```

| Run | Success rate | Mean best IoU | Angle error | Target-prompt gap |
|---|---:|---:|---:|---:|
| `image_only` | 45.47% | 0.2694 | 44.73 deg | 0.0000 |
| `film` | 55.90% | 0.3399 | 35.68 deg | 0.0366 |
| `pcghnet_full_v2` | **57.41%** | **0.3576** | **34.62 deg** | **0.0544** |

## 5. Generate qualitative results

```bash
python visualize.py \
  --checkpoint /root/autodl-tmp/pcghnet_runs/pcghnet_full_v2/best.pt \
  --manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/test.jsonl \
  --output-dir /root/autodl-tmp/pcghnet_runs/pcghnet_full_v2/visualizations \
  --max-samples 50
```

Red is the top prediction, green is ground truth, and the white edge indicates orientation. `index.json` stores prompts and decoded predictions.
