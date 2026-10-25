# Ball model comparison — 25 October 2026

This is **model selection on validation**, using the existing local MLflow
checkpoints. No new model was trained and no annotations were changed.
Independent test matches remain necessary before claiming release accuracy.

## What was checked before inference

The protocol was committed in `3b1e556` before running the comparison. Its
SHA-256 identifier is
`be34895a8d7752b41f908baf1aeb42122ea53e28e7b4638a43b94283bf68c259`.
The evaluator implementation is recorded in `d93e570`; the protocol also hashes
the source files actually used, so a dirty working tree cannot silently change
the scientific recipe.

1. **Inventory the runs.** MLflow contains six completed 30-epoch campaigns,
   one completed two-epoch historical baseline, eight limited-batch smoke tests
   and four incomplete attempts. Three incomplete attempts are soft-deleted in
   MLflow and appear in the supplementary data audit. No completed V4 training
   was found. Smoke tests and incomplete attempts do not enter the ranking.
2. **Identify the actual weights.** Every available `best.pt` and `best_f1.pt`
   from the seven completed campaigns is included: 12 checkpoints. Local files
   were matched by SHA-256 against the MLflow artifacts. `last.pt` is excluded
   by the policy frozen before evaluation; this is a comparison of archived
   training-selected checkpoints, rather than a sweep over all epochs.
3. **Check data and splits.** The six full campaigns share dataset ID
   `b76a9159ef4513585a537c645f75b4200e8c5ef11a1e1d2379d11d4c7b991c83`.
   YOLO, SDK and V3 exports have distinct format-specific export IDs. All three
   exports passed their complete artifact checksum verification. Their validation
   frame records, annotations and JPEG bytes are identical. All 2,929 V3 PNGs
   were also decoded and checked pixel-for-pixel against the common JPEGs.
   Training and validation match identities do not overlap in these exports.
4. **Normalize the task.** Each model produces one ball position per source
   frame. YOLO contributes the center of its most confident class-0 box;
   TrackNet contributes the bounding-rectangle center of the largest contour
   above its heatmap threshold. Both use the same source-coordinate scorer.
5. **Control temporal information.** YOLO uses one frame, V5 uses the last head
   of three consecutive frames, V3 the last head of eight. No future frame or
   overlapping-head average is allowed. Clips and missing exported frames reset
   the context. Reviewed camera-cut annotations are unavailable.
6. **Freeze decisions and measurement.** Heatmap threshold 0.5, YOLO confidence
   0.25, IoU parameter 0.7, 15-source-pixel primary tolerance; sensitivity at
   5/10/15/25 pixels. FP32, batch 1, RTX 5080, four PyTorch CPU threads,
   deterministic PyTorch execution. No threshold was tuned from these results.

The historical baseline has a different training dataset ID
(`6c7036e1533857a4d7f6bc54adcd701e32d68e144c4aa47991d5349c56adbf2c`)
and two training epochs. It is evaluated on the same current validation cohort
as the other models, and is reported as a development reference.

## Common cohort and scoring

| Match | Exported validation frames | Common causal targets |
| --- | ---: | ---: |
| PHX–OKC G4 | 1,507 | 1,344 |
| POR–SAS G5 | 1,422 | 1,283 |
| Total | 2,929 | 2,627 |

The main comparison scores the **same 2,627 frames for every checkpoint**.
These targets have at least eight consecutive exported frames of context and
represent 89.69% of the exported validation frames. This restriction favors
continuous, annotated passages; it is exposed rather than hidden in the score.

Secondary results score each model's own available targets and report coverage
on all 2,929 exported frames: 100% for YOLO, 96.86% for V5, 89.69% for V3.
`coverage_adjusted_f1` additionally counts each uncovered known ball as a false
negative. These metrics distinguish localization quality from context availability.

A prediction within the tolerance is a TP. Abstention on a known ball is an FN.
A displaced prediction is both an FP and an FN. Median/P95 error includes every
localized pair, including displaced predictions; abstentions are represented by
recall, never assigned zero error. Each target is counted once.
Both source clips are 1920 × 1080: the 15-source-pixel tolerance corresponds
to four input pixels at 512 × 288. Coordinates are restored through the export
geometry before computing the distance.

**All exported validation annotations are `visible`.** Unknown positions and
unannotated frames were excluded by the frozen export policy. There are no
reviewed absent-ball negatives and no localizable occlusions in this cohort.
Precision here penalizes wrong positions on positive frames; it does not estimate
the false-alarm rate when the ball is absent. This benchmark cannot validate
robustness to real occlusion.

## Results

| Model | Localization F1 | Precision | Recall | Error median / P95 (px) | Context coverage | Adapter median / P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26L | 0.9451 | 0.9700 | 0.9216 | 3.24 / 8.14 | 100.00% | 5.19 / 5.92 |
| V5 + TOTNet | 0.9219 | 0.9343 | 0.9098 | 4.37 / 188.86 | 96.86% | 18.14 / 18.65 |
| YOLO26M | 0.9211 | 0.9554 | 0.8892 | 3.18 / 9.64 | 100.00% | 4.06 / 4.79 |
| YOLO26S | 0.9109 | 0.9419 | 0.8820 | 3.24 / 173.46 | 100.00% | 3.89 / 4.50 |
| TrackNet V3 | 0.8572 | 0.9630 | 0.7724 | 3.60 / 10.42 | 89.69% | 83.68 / 85.72 |
| TrackNet V5 | 0.8148 | 0.8224 | 0.8074 | 4.45 / 903.85 | 96.86% | 18.18 / 18.60 |
| V5 (2 epochs, historical) | 0.7528 | 0.8125 | 0.7012 | 4.71 / 783.07 | 96.86% | 17.90 / 18.51 |

Each row uses the archived checkpoint with the highest common-cohort validation F1 for that training run. See [all 12 checkpoints](checkpoint-table.md) for the alternatives.

![Quality and adapter latency on the same validation targets](quality-latency.png)

**Development candidate: YOLO26L `best_f1.pt`, epoch 30** from training run `0873327e0d7f402a95ef68fb02d7e626`, evaluated in run `798cb49ed5f64ed1b92b3726fbe5764d`. Its F1 is 0.9451, precision 0.9700 and recall 0.9216. Compared with V5+TOTNet’s selected checkpoint, F1 increases by 0.0233 (2.33 percentage points) and adapter median latency falls from 18.14 to 5.19 ms, about 3.5 times faster. This supports the next integration experiment on these validation matches.

YOLO26S already outperforms the archived V3/V5 checkpoints. YOLO26M is very close to TOTNet in F1 (0.9211 versus 0.9219), with higher precision and lower recall, at roughly a quarter of the adapter latency. YOLO26L improves both precision and recall over TOTNet at the announced thresholds. These observed differences do not establish statistical significance across new matches.

**Pipeline status:** the comparison adapter supports YOLO/V3; the video pipeline still supports V5/V5+TOTNet and its default recipe still references the two-epoch V5. Next, add the YOLO ball adapter, replay the chosen checkpoint on raw video with cut handling, and confirm quality on independently annotated matches. No production registry promotion or full-pipeline performance claim is made here.

### Per-match validation F1

| Model | PHX–OKC G4 (1,344 targets) | POR–SAS G5 (1,283 targets) |
| --- | ---: | ---: |
| YOLO26L | 0.9563 | 0.9335 |
| V5 + TOTNet | 0.9325 | 0.9105 |
| YOLO26M | 0.9250 | 0.9172 |
| YOLO26S | 0.9312 | 0.8895 |
| TrackNet V3 | 0.8717 | 0.8414 |
| TrackNet V5 | 0.8254 | 0.8036 |
| V5 (2 epochs, historical) | 0.7674 | 0.7365 |

### Sensitivity to source-pixel tolerance

Predictions and thresholds stay fixed; only the acceptance radius changes.

| Model | F1 @5 px | F1 @10 px | F1 @15 px | F1 @25 px |
| --- | ---: | ---: | ---: | ---: |
| YOLO26L | 0.7773 | 0.9397 | 0.9451 | 0.9451 |
| V5 + TOTNet | 0.5894 | 0.9014 | 0.9219 | 0.9265 |
| YOLO26M | 0.7681 | 0.9176 | 0.9211 | 0.9215 |
| YOLO26S | 0.7568 | 0.9054 | 0.9109 | 0.9113 |
| TrackNet V3 | 0.6215 | 0.8403 | 0.8572 | 0.8589 |
| TrackNet V5 | 0.5628 | 0.8010 | 0.8148 | 0.8171 |
| V5 (2 epochs, historical) | 0.4904 | 0.7344 | 0.7528 | 0.7540 |

### All exported frames and coverage

| Model | Own covered-cohort F1 | Covered targets / 2,929 | F1 counting uncovered known balls as FN |
| --- | ---: | ---: | ---: |
| YOLO26L | 0.9401 | 2929 / 2,929 | 0.9401 |
| V5 + TOTNet | 0.9215 | 2837 / 2,929 | 0.9066 |
| YOLO26M | 0.9157 | 2929 / 2,929 | 0.9157 |
| YOLO26S | 0.9033 | 2929 / 2,929 | 0.9033 |
| TrackNet V3 | 0.8572 | 2627 / 2,929 | 0.8058 |
| TrackNet V5 | 0.8179 | 2837 / 2,929 | 0.8047 |
| V5 (2 epochs, historical) | 0.7540 | 2837 / 2,929 | 0.7410 |

The common-cohort table remains the main comparison. The own-cohort scores use different frame subsets; coverage-adjusted scores expose missing context under this export policy, rather than raw-stream availability.

### Why training maxima cannot choose the winner

The [historical MLflow training metrics](training-metric-context.json) report a maximum of 0.8943 for V5 and 0.8627 for YOLO26S. The shared causal point-localization evaluation ranks their archived candidates in the opposite order: 0.8148 for V5 versus 0.9109 for YOLO26S. Historical maxima may also belong to different or unsaved epochs. This illustrates the combined effect of metric definitions, temporal support and checkpoint selection; it is not an attribution to one cause.

## Interpretation limits

- Training YOLO box AP/F1 and TrackNet window-head localization F1 measure
  different tasks and support. Their historical maxima cannot establish the
  ranking above. Saved `best.pt` also uses different criteria: validation loss
  for TrackNet and Ultralytics fitness for YOLO. `best_f1.pt` is an additional
  historical criterion, when available. V5 has no archived `best_f1.pt`; its
  peak training-log F1 does not identify a recoverable checkpoint.
- The main table selects the highest **common validation F1** between the
  archived `best` and `best_f1` for each training run, preferring `best` on an
  exact tie. This is another selection on validation, so its scores are
  optimistic for deployment. The complete table preserves all 12 outcomes.
- Localization percentiles are conditional on returning a position. Abstaining
  more often can lower the reported error while reducing recall. P95 also hides
  failures rarer than 5% of localized pairs; use precision/recall, displaced counts,
  mean error and the recorded examples alongside it.
- Confidence scores are not calibrated across families. This compares complete
  model/preprocessing/decoder configurations at the announced thresholds, rather
  than architecture in isolation. YOLO initialization, augmentation and training
  objectives differ. Thirty epochs do not equalize training compute.
- V5+TOTNet names the local V5 training variant with photometric/geometric and
  occlusion augmentations plus a weighted heatmap loss. It uses the same V5
  inference architecture. Its results alone do not identify which training change
  caused an improvement; that would require an ablation.
- There are only two validation matches with strongly correlated neighboring
  frames. No confidence interval treating frames as independent, statistical
  significance claim or independent generalization claim is made.
- Causality is verified on exported sequences. Raw-video context, camera cuts,
  negative examples and end-to-end trajectory stability remain to be tested.
  Missing exported frames may contain usable unannotated input in a real stream.
- Weight provenance does not establish that public pretrained models never saw
  similar source footage. Imported pretraining data has not been independently
  audited for source overlap.

## Inspecting individual failures

Five targeted frames were visually inspected alongside the existing labels.
This is a spot check by Codex, rather than a human review of the full annotation
set. No labels were changed. Local overlays remain under
`release/ball-comparison-review/`; [the recorded case metrics](review-cases.json)
contain source frame indices for reproduction without publishing private footage.

| Case | Source frame | Observation against the current label |
| --- | --- | --- |
| Large displacement | POR–SAS, 875 | Both TOTNet checkpoints predict the opposite side of the court, about 1,401 px from the ball. A good aggregate score still allows severe mistakes. |
| Improvement over V5 | PHX–OKC, 1,289 | On a ball in flight, V5 is 1,635 px away; the two TOTNet checkpoints are within 2 px. |
| Checkpoint tradeoff | POR–SAS, 872 | TOTNet `best` is 7.06 px away; `best_f1` is 1,383 px away on the same frame. Selecting by F1 can worsen the tail of the error distribution. |
| Abstention | PHX–OKC, 233 | TOTNet `best_f1` peaks at 0.4625 and abstains at threshold 0.5; `best` barely passes the threshold and is 14.91 px away. This illustrates sensitivity to a fixed decoder threshold. |
| Winner failure | POR–SAS, 1,114 | YOLO26L predicts the spectator area on the opposite side, 1,590 px from the annotated ball. Its 8.14 px P95 does not rule out rare, severe errors. |

On the first four frames, YOLO26L respectively abstains, localizes within 1.67 px,
5.51 px and 4.40 px. These targeted cases illustrate tradeoffs; they are not a
random sample or another aggregate accuracy estimate.

## Latency scope

Each checkpoint is timed separately after five warmup calls. Thirty evenly
spaced common targets are measured three times (90 observations), synchronizing
CUDA before and after each call. The same target indices are used for all models.
Input windows are already decoded RGB arrays. Timing includes preprocessing,
CPU-to-GPU transfer, forward execution and prediction decoding to input pixels.
It excludes disk I/O, model loading, source-coordinate restoration and all other
pipeline stages. V3/V5 recompute their full window each time. Median/P95 therefore
describes this adapter, rather than full-video throughput or optimized streaming.
Small timing differences between checkpoints of the same architecture can reflect
measurement variation. The epoch-20 YOLO26M `best` and `best_f1` produce identical
prediction files, with recorded medians of 4.06 and 3.95 ms.

## Evidence and reproduction

- [Frozen protocol and active MLflow inventory](protocol.json)
- [All 19 archived checkpoints, including final/resume states](checkpoint-inventory.json)
- [Data audit and soft-deleted incomplete runs](data-audit.json)
- [All checkpoint metrics and evaluation run IDs](results.json)
- [Per-run checkpoint selection](selection.json)
- [Development candidate decision](decision.json)
- [Historical training metric context](training-metric-context.json)
- [Complete checkpoint table](checkpoint-table.md)
- [Clean-source regression checks](checks-summary.json)
- [Post-evaluation integrity and independent score recount](verification.json)
- [Evaluation commands and conventions](../../training/ball/EVALUATION.md)

The local `ball-model-comparison` MLflow experiment stores a separate run per
checkpoint with resolved configuration, environment, source fingerprints,
protocol, metrics and per-frame predictions. Raw predictions and private source
footage remain in local MLflow/run artifacts. Public aggregate JSON files
preserve training/evaluation run IDs and checkpoint checksums. Reproduction needs
the corresponding private exports, local weights, pinned upstream references and
CUDA environment; the repository alone does not include them.
