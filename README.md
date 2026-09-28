# PCGH-Net: Language-Driven Grasp Detection

PCGH-Net predicts a 2D rectangular grasp `(x, y, w, h, theta)` from an RGB image and a natural-language instruction.

## Proposed method

The method has two language-specific components:

1. **Multi-scale prompt modulation.** A frozen MiniLM sentence encoder conditions every level of a ResNet-18 feature pyramid through FiLM before dense grasp prediction.
2. **Low-overlap negative-prompt consistency.** For each sample, the dataset builder selects the same-scene alternative prompt with the lowest grasp-label overlap. A margin loss requires the target prompt to score the target grasp region above this negative prompt.

The network predicts a center heatmap, sub-cell offset, normalized width and height, and `(sin(2 theta), cos(2 theta))`. Positive and negative prompts share the visual backbone computation.

```text
image -> ResNet-18 -> FPN -> FiLM(prompt) -> dense grasp heads
prompt -> frozen MiniLM --------^              center/offset/size/angle
negative prompt -> shared features -> target-region margin loss
```

## Results

The final ablation uses a scene-disjoint 5,000-scene Grasp-Anything++ subset with 11,588/1,413/1,458 train/validation/test samples.

| Method | Success rate | Mean best IoU | Angle error | Target-prompt gap |
|---|---:|---:|---:|---:|
| Image-only | 45.47% | 0.2694 | 44.73 deg | 0.0000 |
| FiLM | 55.90% | 0.3399 | 35.68 deg | 0.0366 |
| PCGH-Net | **57.41%** | **0.3576** | **34.62 deg** | **0.0544** |

A grasp is successful when rotated IoU is at least 0.25 and antipodal angle error is at most 30 degrees. See [EXPERIMENTS.md](EXPERIMENTS.md) for exact preparation, training, evaluation, and visualization commands.

## Repository layout

```text
pcghnet/model.py          ResNet-18 FPN, FiLM fusion, and dense heads
pcghnet/text_encoder.py   frozen MiniLM and offline hash encoder
pcghnet/targets.py        dense grasp-target generation
pcghnet/losses.py         detection and prompt-consistency losses
pcghnet/geometry.py       rotated IoU and grasp success criterion
tools/prepare_ga_pp_subset.py  official subset preparation
train_full.py             full training with validation and resume
evaluate.py               held-out metrics
visualize.py              qualitative overlays
```

## Setup and tests

```bash
pip install -r requirements.txt
python smoke_test.py
python -m unittest discover -s tests -v
```

Python 3.9+ is recommended. For the verified AutoDL environment, use `requirements-autodl.txt` and [AUTODL.md](AUTODL.md).

## Minimal data format

Each JSONL record contains an image path, prompt, grasp list, and optional negative prompt. Coordinates are in image pixels and angles are in radians.

```json
{"image":"images/example.jpg","prompt":"grasp the red mug","negative_prompt":"grasp the bowl","grasps":[{"x":208,"y":173,"w":92,"h":34,"theta":0.18}]}
```

## Inference

```bash
python infer.py \
  --checkpoint outputs/pcghnet/best.pt \
  --image example.jpg \
  --prompt "grasp the red mug"
```

Dataset: [Grasp-Anything++](https://airvlab.github.io/grasp-anything/docs/grasp-anything-pp/)
