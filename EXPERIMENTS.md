# Final experiment recipe

All three runs use the same scene-disjoint 80/10/10 split, ImageNet-pretrained
ResNet-18, frozen MiniLM text features where applicable, seed, optimizer, and
training schedule. Only the language-conditioning component changes.

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

Do not download or unzip the 65 GB image archive. The preparation script reads
only the selected JPEG byte ranges from the official split archive.

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
cat /root/autodl-tmp/grasp_anything_pp/subset_5k/subset_stats.json
```

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
  --output-dir /root/autodl-tmp/pcghnet_runs/pcghnet_full
```

Run these sequentially on one RTX 4090. Keep `best.pt` selected by validation
success rate; never select a checkpoint using the test split.

## 4. Final test metrics

```bash
for RUN in image_only film pcghnet_full; do
  python evaluate.py \
    --checkpoint /root/autodl-tmp/pcghnet_runs/$RUN/best.pt \
    --manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/test.jsonl \
    --batch-size 32 \
    --output /root/autodl-tmp/pcghnet_runs/$RUN/test_metrics.json
done
```

Report success rate, mean best rotated IoU, mean angle error, and target prompt
gap. The image-only prompt gap should be exactly zero by construction.

## 5. Qualitative results

```bash
python visualize.py \
  --checkpoint /root/autodl-tmp/pcghnet_runs/pcghnet_full/best.pt \
  --manifest /root/autodl-tmp/grasp_anything_pp/subset_5k/test.jsonl \
  --output-dir /root/autodl-tmp/pcghnet_runs/pcghnet_full/visualizations \
  --max-samples 50
```

Red is the top prediction, green is ground truth, and the white edge indicates
the grasp rectangle orientation. `index.json` stores each prompt and decoded
prediction for figure selection.
