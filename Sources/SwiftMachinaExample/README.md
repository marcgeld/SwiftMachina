# SwiftMachinaExample

An end-to-end walkthrough of the SwiftMachina API on a real dataset: load a CSV with TabularData, split it, standardize it, then train, evaluate, and time all eleven classifiers.

## What It Does

1. **Loads the bundled Wisconsin breast-cancer dataset** (`Resources/breast_cancer.csv`, 569 samples × 30 features) with `TabularData.DataFrame`, mapping the diagnosis column (`M`/`B`) to binary labels.
2. **Splits it** with `trainTestSplit` (80/20, stratified, `randomState: 42`).
3. **Trains every estimator** — LogisticRegression, SVM, KNN, GaussianNaiveBayes, LDA, QDA, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, and XGBoostClassifier — each wrapped in a `Pipeline` with a `StandardScaler` front stage. SVM gets its labels remapped to `{-1, +1}` as its hinge loss expects.
4. **Reports per-model metrics** from `ConfusionMatrix`: accuracy, precision, recall, F1, balanced accuracy, specificity, MCC, the raw TP/TN/FP/FN counts, and wall-clock train/inference times.
5. **Writes `swift_metrics.csv`** to the working directory — one row per model — so an external benchmark harness can compare SwiftMachina against scikit-learn or other libraries. Cross-library comparison intentionally lives outside this repository (see the main README's rules).

For the small synthetic per-model walkthroughs, including JSON fitted-state save/load round trips, see [SwiftMachinaMiniExamples](../SwiftMachinaMiniExamples) instead — one file per estimator.

## Running

Command-line SwiftPM cannot build MLX's Metal shaders, so use Xcode's build system (see "Building And Testing" in the [main README](../../README.md)):

```bash
xcodebuild -scheme SwiftMachinaExample -destination 'platform=macOS,arch=arm64' build
```

Then run the built product from `DerivedData`, or run the scheme directly in Xcode. `swift build` compiles the target fine but the executable crashes at runtime with `Failed to load the default metallib` — that is a documented mlx-swift limitation, not a bug.

## Output

For each model:

```text
--- XGBoost ---
  Accuracy:         0.9737
  Precision:        0.9524
  ...
  TrainTime:        0.1234s
  InferenceTime:    0.0021s
  TP: 40  TN: 71  FP: 2  FN: 1
```

followed by `swift_metrics.csv` with the columns:

```text
Model,Status,Accuracy,Precision,Recall,F1,BalancedAccuracy,Specificity,MCC,AUC,TrainTimeSec,InferenceTimeSec,TN,FP,FN,TP
```

(The AUC column is left empty until `predictProba` is available on every classifier.)

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset is bundled as a package resource, so nothing needs downloading to run the example. To refresh or re-fetch it:

```bash
curl -sSLO https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin/resolve/main/breast_cancer.csv
```

References:

- [Hugging Face (stable, clean CSV)](https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin/blob/main/breast_cancer.csv)
- [GitHub (raw CSV)](https://github.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/data.csv)
- [UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
