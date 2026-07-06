# SwiftMachina

SwiftMachina is where mathematics meets machine intelligence. Built on [MLX](https://github.com/ml-explore/mlx-swift) and [SwiftNumerica](https://github.com/marcgeld/SwiftNumerica), optimized for Apple Silicon, it embraces the Renaissance ideal that science, engineering and art are all expressions of the same human curiosity.

It is a classical machine-learning framework in the spirit of scikit-learn — estimators with `fit`/`predict`, transformers, pipelines, and metrics — expressed in idiomatic Swift: value types, protocol-oriented design, Swift Package Manager, and Swift Testing.

## Design Philosophy

The fundamental data type is `MLXArray` (Float32). All dataset-sized computation — means, variances, matrix products, gradients, distance matrices — runs through MLX so it stays vectorized and lazy on Apple Silicon. Training loops use MLXNN and MLXOptimizers.

Two deliberate exceptions leave the MLX graph:

- **Tree building** (`DecisionTree` and the ensembles built on it) is inherently branchy, sequential work and is implemented in plain Swift on `[Float]` buffers.
- **Small-matrix linear algebra** (features × features covariance work in `LDA`/`QDA`) goes through SwiftNumerica's LAPACK-backed routines in **Double precision** via the internal `NumericaBridge`. MLX would run these on the CPU stream anyway, and Double is strictly better for covariance inversion. MLX Float32 ops remain as fallback for degenerate input.

The correctness baseline is **scikit-learn**: the test suite includes parity tests that compare estimator behavior against sklearn results on the breast-cancer dataset. The full sklearn benchmark suite lives in a separate repository.

The public API mirrors scikit-learn conventions, adapted to Swift value semantics: estimators are `struct`s, `fit` is `mutating`, and there are no reference-type model objects.

## What's Implemented

- **Estimators**: `LogisticRegression` (MLXNN + SGD), `SVM` (linear, hinge loss), `KNN` (vectorized distance matrix), `DecisionTree` (CART, gini), `RandomForest`, `ExtraTrees`, `GradientBoosting`, `GaussianNaiveBayes`, `LDA`, `QDA`
- **Preprocessing**: `StandardScaler`, `trainTestSplit` (stratified, seeded — mirrors sklearn's `train_test_split(random_state:stratify:)`)
- **Pipeline**: chains transformers and a final model behind one `fit`/`predict`
- **Metrics**: `Accuracy`, `ConfusionMatrix` (accuracy, precision, recall, F1, specificity, balanced accuracy, MCC)
- **Losses**: `BinaryCrossEntropy` (with-logits and with-probabilities variants)
- **Core**: `Estimator`/`Predictor`/`Classifier`/`Regressor`/`Transformer` protocols, `SeededRandomNumberGenerator` (SplitMix64) for reproducibility

## Example Usage

```swift
import MLX
import SwiftMachina

// X: [N, features], y: [N, 1] — Float32 MLXArrays

// Stratified, seeded 80/20 split
let split = trainTestSplit(X: X, y: y, testSize: 0.2, randomState: 42, stratify: true)

// Standardize + classify in one pipeline
var pipeline = Pipeline(steps: [
    .transformer(StandardScaler()),
    .model(LogisticRegression(inputSize: X.shape[1]))
])
pipeline.fit(X: split.Xtrain, y: split.ytrain)
let predictions = pipeline.predict(X: split.Xtest)

// Evaluate
let cm = ConfusionMatrix().compute(split.ytest, predictions)
print("accuracy: \(cm.accuracy), F1: \(cm.f1), MCC: \(cm.mcc)")
```

A complete, runnable version of this workflow (CSV loading via TabularData, all ten estimators compared, timings) lives in [Sources/SwiftMachinaExample](Sources/SwiftMachinaExample).

## Package Structure

```text
Sources/
├── SwiftMachina
│   ├── Core          (protocols, shared types, NumericaBridge)
│   ├── Estimators    (LogisticRegression, SVM, KNN, DecisionTree,
│   │                  RandomForest, ExtraTrees, GradientBoosting,
│   │                  GaussianNaiveBayes, LDA, QDA)
│   ├── Preprocessing (StandardScaler, TrainTestSplit)
│   ├── Pipeline
│   ├── Metrics       (Accuracy, ConfusionMatrix)
│   └── Losses        (BinaryCrossEntropy)
├── SwiftMachinaExample     (executable: breast-cancer walkthrough)
└── SwiftMachinaBenchmarks  (executable: synthetic-data benchmarks)
Tests/
└── SwiftMachinaTests       (Swift Testing: unit + sklearn parity tests)
```

## Requirements

- Swift tools 6.3 (Xcode 26) or newer.
- macOS 26 / iOS 26 or newer, per `Package.swift`.
- Dependencies: `mlx-swift` 0.31.3+, `SwiftNumerica` 0.1.0+ (resolved automatically).

## Building And Testing

```bash
# Compiles everything (library, example, benchmarks, tests)
swift build
```

**Running requires Xcode's build system.** Command-line SwiftPM cannot build MLX's Metal shaders, so `swift test` and `swift run` compile fine but crash at runtime with `Failed to load the default metallib`. This is a documented mlx-swift limitation, not a bug in this package. Use `xcodebuild`:

```bash
# Run the test suite (unit + parity tests)
xcodebuild test -scheme SwiftMachina-Package -destination 'platform=macOS,arch=arm64'

# Run the example / benchmarks
xcodebuild -scheme SwiftMachinaExample -destination 'platform=macOS,arch=arm64' build
xcodebuild -scheme SwiftMachinaBenchmarks -destination 'platform=macOS,arch=arm64' build
```

## SwiftNumerica Integration

`Sources/SwiftMachina/Core/NumericaBridge.swift` is the single point of contact with SwiftNumerica. It converts small `MLXArray` matrices to SwiftNumerica's `Matrix` (Double) and exposes:

- `numericaInverse(_:)` — LAPACK matrix inversion, used for covariance matrices in `LDA.fit` and `QDA.predict`.
- `numericaLogDeterminant(symmetric:)` — log-determinant via the sum of log-eigenvalues (stable where a direct determinant would under- or overflow). The input is symmetrized first because Float32 covariance products carry last-bit asymmetry that SwiftNumerica's symmetric eigensolver rejects.

Both return `nil` on degenerate input, and callers fall back to the original MLX CPU ops. This scope is intentional — see the agent rules below.

## Continuous Integration

`.github/workflows/build.yml` runs on pushes and pull requests to `main`:

- `swift build` with a cached `.build` keyed on `Package.resolved`.
- `xcodebuild build-for-testing`, which also compiles the Metal shaders and the test bundle.

Tests are **compiled but not executed** in CI: GitHub-hosted macOS runners have no usable GPU for MLX (mlx-swift runs its own macOS tests on self-hosted runners for the same reason). Run the test suite locally before pushing.

## Using SwiftMachina From Another Package

```swift
// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "ExampleProject",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/marcgeld/SwiftMachina.git", branch: "main")
    ],
    targets: [
        .target(
            name: "ExampleProject",
            dependencies: [
                .product(name: "SwiftMachina", package: "SwiftMachina")
            ]
        )
    ],
    swiftLanguageModes: [.v6]
)
```

The consuming app must be built with Xcode / `xcodebuild` for the MLX Metal shaders to be available at runtime.

## Future Plans

Roughly in priority order:

- **XGBoost-style gradient boosting**: regularized second-order boosting (gradient + hessian leaf weights, L1/L2 penalties, shrinkage, column/row subsampling), histogram-based split finding, and early stopping — as a separate estimator alongside the current `GradientBoosting`.
- **Regression support**: put the `Regressor` protocol to work — `LinearRegression`, `DecisionTreeRegressor`, and regression variants of the ensembles, plus regression metrics (MSE, MAE, R²).
- **Multiclass beyond the discriminant models**: `KNN`, the confusion matrix, and the losses are binary today; extend to multiclass (one-vs-rest where natural) and add a multiclass confusion matrix.
- **Model selection**: k-fold cross-validation and grid search — the `Hyperparameter` container in Core already reserves the API surface.
- **Probability everywhere**: `predictProba` on all classifiers (only `LogisticRegression` has it), enabling ROC-AUC and log-loss metrics.
- **Model persistence**: save/load fitted models (Codable parameters).
- **SwiftNumerica 0.1.1 adoption**: switch the QDA log-determinant to Cholesky and move `SeededRandomNumberGenerator` upstream once SwiftNumerica ships them.

## Contribution Guidelines

- Keep `MLXArray` as the data type of every public `fit`/`predict`/`transform` signature.
- Estimators are value types; `fit` is `mutating`; no reference-type model objects.
- Follow the sklearn naming conventions already in place (`fit`, `predict`, `predictProba`, `fitTransform`, `randomState`, `testSize`).
- Every estimator change must keep the parity tests green — they are the ground truth for numerical behavior.
- Randomized code must accept an optional seed and produce identical results for identical seeds (`SeededRandomNumberGenerator`).
- Use Swift Testing (`@Test`, `#expect`) for new coverage.
- Update this README whenever architecture, modules, or design decisions change.

## Rules For Future Contributors And AI Agents

README.md is the authoritative architecture document. If implementation and README disagree, update the implementation to match README unless explicitly instructed otherwise.

- **Never verify with `swift test` or `swift run`** — they crash at runtime on the missing metallib (see Building And Testing). Always verify with `xcodebuild test -scheme SwiftMachina-Package -destination 'platform=macOS,arch=arm64'`. Do not treat the metallib crash as a code bug and do not try to "fix" it in this package.
- **Dataset-sized math stays on MLX.** Do not route per-sample or per-feature-column computation through SwiftNumerica, CPU loops, or Accelerate — it would leave the GPU and break the lazy-evaluation model. The only sanctioned CPU exceptions are tree building and `NumericaBridge`.
- **SwiftNumerica is for small features × features matrices only**, accessed exclusively through `NumericaBridge`. Widen that bridge only when SwiftNumerica gains a capability this package needs (e.g. Cholesky); do not scatter direct `SwiftNumerica` imports across estimators.
- **The parity tests define correctness.** A change that alters predictions must be justified against scikit-learn behavior, not just compile.
- **Do not add dataset/file-format assumptions to the library.** CSV/TabularData loading belongs in examples and consuming apps, never in `Sources/SwiftMachina`.
- The Python/sklearn benchmark suite lives in a separate repository — do not re-add Python tooling here.
