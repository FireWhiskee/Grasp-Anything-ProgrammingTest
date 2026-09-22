# PCGH-Net: Language-Driven Grasp Detection

This repository is an initial implementation for the Language-Driven Grasp Detection programming test. Given an RGB image and a natural-language prompt, the model predicts a 2D rectangular grasp `(x, y, w, h, theta)` for the referred object.

## Proposed method

**PCGH-Net (Prompt-Conditioned Grasp Heatmap Network)** is a lightweight dense grasp detector with two method-specific components:

1. **Multi-scale prompt modulation.** A small text encoder maps the prompt to one vector. Text-driven FiLM layers modulate every level of a ResNet-18 feature pyramid before the features are fused.
2. **Negative-prompt consistency.** During training, the same image is paired with a wrong prompt. A margin loss requires the correct prompt to produce a stronger response at the target grasp region than the wrong prompt. This explicitly discourages an image-only shortcut.

Positive and negative prompts share one visual-backbone pass, so the language-specific supervision adds only the text/fusion/head computation.

The decoder predicts a grasp-center heatmap, sub-cell center offsets, normalized width/height, and `(sin(2 theta), cos(2 theta))`. The doubled-angle representation handles the 180-degree symmetry of parallel-jaw grasps.

```text
image -> ResNet-18 -> multi-scale FPN -> FiLM(prompt) -> dense grasp heads
prompt -> frozen MiniLM --------------------^          | center, offset, size, angle
wrong prompt -> same network -> ranking consistency loss
```

This is the proposed method rather than only an assembly of standard components: the central hypothesis is that multi-scale prompt modulation plus target-region negative-prompt ranking produces more language-sensitive grasp predictions. The intended ablation is image-only vs. FiLM vs. FiLM plus consistency loss.

## Repository layout

```text
pcghnet/model.py          PCGH-Net, feature pyramid, and FiLM fusion
pcghnet/text_encoder.py   frozen MiniLM and offline hash encoder
pcghnet/targets.py        dense heatmap/size/angle target generation
pcghnet/losses.py         detection and negative-prompt losses
pcghnet/dataset.py        JSONL manifest loader and geometry transforms
pcghnet/decode.py         dense output to rectangular grasps
train.py                  training entry point
infer.py                  single-image inference entry point
smoke_test.py             offline forward/loss/backward/decode check
tests/test_core.py        core unit tests
```

## Setup

Python 3.9+ is recommended. The core code also runs on Python 3.7 with PyTorch 1.13.

```bash
pip install -r requirements.txt
pip install transformers  # required for the recommended MiniLM encoder
```

Run the offline check without downloading data or model weights:

```bash
python smoke_test.py
python -m unittest discover -s tests -v
```

## Data format

Convert Grasp-Anything++ annotations to one JSON object per line. Paths are relative to the manifest. Angles are in radians.

```json
{"image":"images/example.jpg","prompt":"grasp the red mug","negative_prompt":"grasp the bowl","grasps":[{"x":208,"y":173,"w":92,"h":34,"theta":0.18}]}
```

`negative_prompt` is optional. If absent, the loader samples a prompt from another item. See `examples/manifest.example.jsonl`.

## Train and infer

The recommended experiment uses frozen MiniLM and a pretrained ResNet-18:

```bash
python train.py --manifest data/train.jsonl --text-encoder minilm --batch-size 16
python infer.py --checkpoint outputs/last.pt --image example.jpg --prompt "grasp the red mug"
```

Use `--text-encoder hash` for dependency-free development or as a small-text-encoder ablation. Use `--no-negative-prompts` for the key loss ablation.

## Planned evaluation

- Split by scene/image to avoid prompt variants of one image leaking across splits.
- Report rectangle success when rotated IoU is above 0.25 and angle error is below 30 degrees.
- Compare image-only, concatenation/FiLM, and full PCGH-Net.
- Report parameter count, training subset size, and qualitative prompt-swap examples.

The code is implemented but not trained yet, as requested. A 12-16 GB GPU is sufficient at 224 px with batch size 8-16; 24 GB is comfortable for larger batches.

## Dataset

- [Grasp-Anything++ project and download instructions](https://airvlab.github.io/grasp-anything/docs/grasp-anything-pp/)
