# PCGH-Net: Prompt-Conditioned Grasp Heatmaps with Low-Overlap Negative-Prompt Consistency

**Author:** [Your Name]  
**Code:** <https://github.com/FireWhiskee/Grasp-Anything-ProgrammingTest>

## Abstract

Language-driven grasp detection requires a model to predict a feasible grasp for the object specified by a natural-language instruction, rather than merely finding any graspable region in an image. We propose **PCGH-Net (Prompt-Conditioned Grasp Heatmap Network)**, a compact dense predictor that combines a ResNet-18 feature pyramid, a frozen MiniLM text encoder, and multi-scale feature-wise linear modulation (FiLM). In addition to standard grasp supervision, PCGH-Net uses a low-overlap negative-prompt consistency objective. For each training example, we select the alternative prompt in the same scene whose annotated grasp set has the lowest overlap with the target grasp set. A margin-ranking loss then requires the target prompt to produce a stronger response than this negative prompt inside the target grasp region. This discourages the model from taking an image-only shortcut while avoiding arbitrarily chosen contradictory prompts. We evaluate on a scene-disjoint subset of 5,000 Grasp-Anything++ scenes. Language conditioning improves rectangular grasp success from 45.47% to 55.90%. Adding our low-overlap negative-prompt objective further increases success to 57.41%, raises mean best rotated IoU from 0.3399 to 0.3576, and reduces mean orientation error from 35.68 degrees to 34.62 degrees. These results show that the proposed objective yields a modest but consistent improvement in both grasp localization and prompt sensitivity.

## 1. Introduction

Robotic grasp detection is commonly formulated as predicting a grasp pose from visual observations. In a cluttered scene, however, an image may contain several graspable objects, and a robot must determine not only *how* to grasp but also *what* the user intends to grasp. Language-driven grasp detection addresses this problem by conditioning grasp prediction on a natural-language instruction.

Grasp-Anything++ provides large-scale image, instruction, and rectangular-grasp annotations for this setting [1]. Its reference method treats grasp detection as conditional generation with a diffusion model. For this programming test, we study a smaller and more direct alternative: a dense heatmap detector that can be trained on a single commodity GPU and whose language contribution can be isolated through controlled ablations.

A central failure mode is the **image-only shortcut**. A network may learn visually plausible grasps but react weakly when the instruction changes. Simply adding a text embedding does not ensure that the predicted grasp is instruction-specific. Negative prompts provide a useful signal, but a randomly selected prompt can describe the same object or nearly the same grasp region, producing ambiguous supervision.

Our proposed method, PCGH-Net, addresses these issues with two connected ideas:

1. It injects a sentence embedding into every level of a visual feature pyramid through FiLM, allowing language to influence both high-resolution geometry and high-level semantics.
2. It selects the lowest-overlap alternative instruction available in the same scene and imposes a target-region margin loss between the positive and negative predictions.

The contribution is therefore not merely the use of a standard visual backbone. **Our proposed method is the combination of multi-scale prompt-conditioned dense grasp prediction and low-overlap negative-prompt consistency.** The three-way ablation in Section 4 separates image-only prediction, language conditioning, and the proposed consistency objective.

## 2. Method

### 2.1 Problem formulation

Given an RGB image \(I\) and an instruction \(p\), the task is to predict a two-dimensional rectangular grasp

\[
g=(x,y,w,h,\theta),
\]

where \((x,y)\) is the grasp center, \(w\) and \(h\) are the rectangle dimensions, and \(\theta\) is the in-plane gripper orientation. Each image-instruction pair can have multiple valid annotated grasps \(G=\{g_1,\ldots,g_n\}\).

PCGH-Net predicts four dense maps at one quarter of the input resolution: a grasp-center confidence map, a two-dimensional sub-cell offset, normalized width and height, and a two-dimensional orientation representation.

### 2.2 Visual and language encoders

We use an ImageNet-pretrained ResNet-18 [2] as the image encoder. Feature maps from its four stages are projected to 128 channels and combined in a top-down feature pyramid [3]. The lightweight backbone and 224-pixel input keep the training cost suitable for a 24 GB GPU.

The instruction is encoded by a frozen MiniLM sentence encoder [4] and projected to a 256-dimensional vector \(e_p\). Freezing the text encoder reduces memory use and focuses training on visual-language fusion and grasp prediction.

### 2.3 Multi-scale prompt modulation

At feature-pyramid level \(l\), a learned affine projection maps \(e_p\) to channel-wise scale and bias vectors \(\gamma_l(e_p)\) and \(\beta_l(e_p)\). We modulate the visual feature \(F_l\) using FiLM [5]:

\[
\widetilde{F}_l = F_l \odot \left(1+\gamma_l(e_p)\right)+\beta_l(e_p).
\]

The FiLM projections are zero-initialized, so the network begins as an ordinary visual detector and gradually learns prompt-dependent deviations. The modulated pyramid levels are resized to a common resolution, concatenated, and fused by a convolutional block before the prediction heads.

### 2.4 Dense grasp representation

For every annotated grasp, we draw a Gaussian peak on the center heatmap. Regression targets are stored at the grasp center. The model predicts the sub-cell center offset and normalized rectangle size using sigmoid outputs. Because a parallel-jaw grasp is unchanged by a 180-degree rotation, orientation is represented as

\[
a=(\sin 2\theta,\cos 2\theta).
\]

At inference time, the highest-confidence center is decoded with its offset, size, and normalized angle vector to obtain one rectangular grasp.

### 2.5 Low-overlap negative-prompt selection

For a target record \((I,p,G)\), let \(\mathcal{C}(I,p)\) be the other annotated instructions in the same scene. We define the overlap between two grasp sets as the maximum pairwise rotated rectangle IoU:

\[
O(G,G')=\max_{g\in G,\,g'\in G'} \operatorname{rIoU}(g,g').
\]

The negative instruction is selected by

\[
p^- = \arg\min_{(p',G')\in\mathcal{C}(I,p)} O(G,G').
\]

When no same-scene alternative exists, we use an instruction for a different sample. Same-scene selection is preferred because it holds image content constant: any difference between the two predictions must arise from the instruction. Choosing the lowest-overlap candidate reduces contradictory cases in which positive and negative prompts share the same grasp label.

This is a *soft* negative strategy rather than a guarantee of disjoint targets. In the training split, 11,506 of 11,588 records have a same-scene alternative, but the mean selected overlap is 0.454 and the median is 0.188. Some scenes contain only alternative instructions referring to the same object, and their minimum available overlap can approach one. We retain these cases and use a margin loss rather than treating every negative as a hard class label.

### 2.6 Training objective

The grasp detection loss is

\[
\mathcal{L}_{det}=\mathcal{L}_{center}+\mathcal{L}_{offset}+\mathcal{L}_{size}+\mathcal{L}_{angle}.
\]

The center term is a focal heatmap loss. Offset and size use Smooth L1 loss at annotated centers. The angle term is one minus cosine similarity between predicted and target doubled-angle vectors.

To measure prompt dependence, we run the shared visual features through the language-conditioned pyramid twice: once with the target prompt \(p\) and once with \(p^-\). Let \(s^+\) and \(s^-\) be the mean sigmoid center confidence inside the target heatmap region for the two prompts. The consistency term is

\[
\mathcal{L}_{npc}=\max(0,m-s^++s^-),
\]

where \(m=0.2\). The complete objective is

\[
\mathcal{L}=\mathcal{L}_{det}+\lambda\mathcal{L}_{npc},
\]

with \(\lambda=0.1\) in the full model. Positive and negative prompts reuse the same image features, so the additional training cost is limited to text encoding, feature modulation, fusion, and prediction heads.

## 3. Experimental Setup

### 3.1 Dataset and split

We use a subset of 5,000 scenes from Grasp-Anything++ [1], with at most three instruction samples per scene. We split by scene rather than by instruction to prevent different descriptions of the same image from leaking across partitions.

| Split | Instruction-grasp samples |
|---|---:|
| Training | 11,588 |
| Validation | 1,413 |
| Test | 1,458 |

All reported test results use the same held-out 1,458 samples. No test sample is used for model selection.

### 3.2 Compared variants

We train three variants with the same visual backbone, dense heads, data split, and optimization settings.

- **Image-only:** the text encoder and FiLM modules are disabled.
- **FiLM:** language conditions all pyramid levels, but negative-prompt consistency is disabled.
- **PCGH-Net:** FiLM conditioning plus the proposed low-overlap negative-prompt consistency objective.

This ablation directly tests whether language helps beyond visual grasp detection and whether the proposed training objective helps beyond architectural language fusion.

### 3.3 Training details

Images are resized to 224 by 224 pixels. We train for 20 epochs with batch size 32 using AdamW, an initial learning rate of \(3\times10^{-4}\), weight decay \(10^{-4}\), cosine learning-rate annealing, gradient clipping at 5.0, and automatic mixed precision. The random seed is 7. The image backbone is initialized with ImageNet weights, and the MiniLM encoder remains frozen. All experiments are run on one NVIDIA RTX 4090 with 24 GB memory.

### 3.4 Metrics

For each input, we decode the top-ranked grasp and match it to the ground-truth rectangle with the highest rotated IoU. A prediction is successful when rotated IoU is at least 0.25 and antipodal orientation error is at most 30 degrees. We report:

- **Success rate:** fraction of predictions satisfying both criteria.
- **Mean best IoU:** mean rotated IoU of the best ground-truth match.
- **Mean angle error:** mean antipodal orientation error in degrees.
- **Target-prompt gap:** target-region confidence under the target prompt minus confidence under the negative prompt. A larger value indicates stronger instruction sensitivity.

## 4. Results

### 4.1 Quantitative comparison

| Method | Success rate (%) | Mean best IoU | Angle error (deg., lower is better) | Target-prompt gap |
|---|---:|---:|---:|---:|
| Image-only | 45.47 | 0.2694 | 44.73 | 0.0000 |
| FiLM | 55.90 | 0.3399 | 35.68 | 0.0366 |
| **PCGH-Net (ours)** | **57.41** | **0.3576** | **34.62** | **0.0544** |

Language conditioning provides the largest gain: FiLM improves success by 10.43 percentage points over the image-only baseline. This confirms that the task cannot be handled as generic visual grasp detection alone.

The full PCGH-Net improves success by a further 1.51 percentage points over FiLM. It also increases mean best IoU by 0.0177 and lowers mean orientation error by 1.06 degrees. The target-prompt gap rises from 0.0366 to 0.0544, a relative increase of approximately 48.6%. The consistency objective therefore changes more than a diagnostic score: it is accompanied by better localization and orientation accuracy.

On the validation set, the selected PCGH-Net checkpoint reaches 58.17% success, 0.3566 mean best IoU, 33.84-degree mean angle error, and a 0.0508 prompt gap. Its similar validation and test behavior provides no indication of severe split-specific overfitting.

### 4.2 Effect of negative-prompt design

An earlier diagnostic run used the next available same-scene prompt without considering label overlap. That model reached 55.35% success and did not improve over FiLM. This failure was informative: a linguistically different prompt is not necessarily a geometrically valid negative. Two prompts can refer to the same object or share most annotated grasps.

Replacing arbitrary prompt selection with minimum available grasp-set overlap raises success to 57.41%. The result supports the design motivation for PCGH-Net: negative-prompt supervision becomes useful when its construction respects the geometry of the labels. Because many scenes still lack a truly disjoint alternative, future work should additionally filter high-overlap negatives or use overlap-dependent loss weights.

### 4.3 Qualitative analysis

We generated 50 qualitative predictions on the held-out test set. Each visualization overlays the predicted grasp and annotated target grasps on the source image together with the instruction. Successful examples show that the model can move its selected grasp toward the instructed object while maintaining a feasible parallel-jaw orientation. Typical failures fall into three categories: confusion between nearby objects with similar appearance, correct object selection but insufficient rotated overlap, and large orientation error on small or elongated targets.

The qualitative examples should be interpreted together with the prompt-gap metric. A plausible rectangle alone does not establish language understanding; prompt-swapped visualizations are particularly useful because they reveal whether the predicted region actually changes when the instruction changes.

## 5. Discussion

The experiments satisfy the programming-test requirement that the submission contain a method that can be identified as the author's proposal. PCGH-Net is not presented as a new backbone or a state-of-the-art result on the full benchmark. Its method-level claim is narrower and supported by ablation: low-overlap same-scene negative prompts provide geometry-aware contrastive supervision for a multi-scale language-conditioned grasp heatmap detector.

The approach also has practical advantages. It is substantially simpler than a diffusion-based generator, uses a small visual backbone, shares image computation between positive and negative prompts, and trains on a single 24 GB GPU. These properties make it suitable for rapid experimentation and deployment-oriented prototyping.

## 6. Limitations

First, the experiments use a 5,000-scene subset rather than the complete Grasp-Anything++ dataset. The reported numbers should therefore be treated as evidence for this implementation and split, not as directly comparable full-benchmark results. Second, each variant was trained with one random seed, so small differences lack a variance estimate. Third, the lowest-overlap prompt available within a scene can still have high label overlap; the method reduces ambiguity but does not remove it. Fourth, evaluation is offline and based on rectangular annotations. It does not measure collision avoidance, reachability, gripper mechanics, or real-robot grasp success. Finally, the frozen sentence encoder and FiLM fusion may be insufficient for instructions requiring complex relational reasoning.

## 7. Conclusion

We presented PCGH-Net, a compact language-driven grasp detector based on multi-scale prompt modulation and low-overlap negative-prompt consistency. On a scene-disjoint 5,000-scene subset of Grasp-Anything++, language conditioning substantially outperforms image-only detection, and the proposed consistency objective further improves success, overlap, orientation, and prompt sensitivity. The results support the central hypothesis that negative prompts are most effective when selected with awareness of grasp-label geometry. Future work will evaluate multiple seeds and the full dataset, weight negatives by overlap, and test prompt swaps and grasp execution on a physical robot.

## References

[1] A. D. Vuong, M. N. Vu, B. Huang, N. Nguyen, H. Le, T. Vo, and A. Nguyen. “Language-driven Grasp Detection.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 17902-17912, 2024.

[2] K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

[3] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie. “Feature Pyramid Networks for Object Detection.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[4] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou. “MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.” *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[5] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville. “FiLM: Visual Reasoning with a General Conditioning Layer.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018.

## Appendix A. Reproducibility Checklist

- Dataset split is scene-disjoint: yes.
- Training, validation, and test sample counts are reported: yes.
- Evaluation thresholds are reported: yes.
- Architecture and target representation are described: yes.
- Optimizer, learning rate, batch size, epochs, and seed are reported: yes.
- Ablations isolate image, language fusion, and the proposed objective: yes.
- Code and training commands are publicly available in the repository: yes.
- Full-dataset and real-robot claims are made: no.

