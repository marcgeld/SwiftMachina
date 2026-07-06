import Testing
import MLX
@testable import SwiftMachina

// MARK: - Test Data

/// Linearly separable binary dataset for model tests.
/// Class 0: features around (-2, -2), Class 1: features around (+2, +2)
private func makeLinearData(n: Int = 100) -> (X: MLXArray, y: MLXArray) {
    let half = n / 2

    var features: [Float] = []
    var labels: [Float] = []

    for i in 0..<half {
        let offset = Float(i) * 0.02
        features.append(contentsOf: [-2.0 + offset, -2.0 + offset])
        labels.append(0)
    }
    for i in 0..<half {
        let offset = Float(i) * 0.02
        features.append(contentsOf: [2.0 + offset, 2.0 + offset])
        labels.append(1)
    }

    let X = MLXArray(features).reshaped([n, 2])
    let y = MLXArray(labels).reshaped([n, 1])
    return (X, y)
}

private func splitData(_ X: MLXArray, _ y: MLXArray, trainRatio: Double = 0.8)
    -> (xTrain: MLXArray, yTrain: MLXArray, xTest: MLXArray, yTest: MLXArray)
{
    let n = X.shape[0]
    let split = Int(Double(n) * trainRatio)
    return (X[0..<split], y[0..<split], X[split..<n], y[split..<n])
}

private func floatValues(_ array: MLXArray) -> [Float] {
    array.asType(.float32).asArray(Float.self)
}

struct CPUDeviceTrait: SuiteTrait, TestTrait, TestScoping {
    func provideScope(
        for test: Test,
        testCase: Test.Case?,
        performing function: @Sendable () async throws -> Void
    ) async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await function()
        }
    }
}

// MARK: - StandardScaler Tests

@Suite("StandardScaler", CPUDeviceTrait())
struct StandardScalerTests {

    @Test func fitTransformProducesZeroMeanUnitVariance() {
        let X = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as [Float]).reshaped([3, 2])
        var scaler = StandardScaler()
        let Xt = scaler.fitTransform(X: X)

        let mean = Xt.mean(axis: 0)
        let std = sqrt(((Xt - mean) * (Xt - mean)).mean(axis: 0))

        let meanValues = mean.asArray(Float.self)
        let stdValues = std.asArray(Float.self)

        for m in meanValues {
            #expect(abs(m) < 1e-5, "Mean should be ~0, got \(m)")
        }
        for s in stdValues {
            #expect(abs(s - 1.0) < 1e-3, "Std should be ~1, got \(s)")
        }
    }

    @Test func inverseTransformRecoversOriginal() {
        let X = MLXArray([10.0, 20.0, 30.0, 40.0] as [Float]).reshaped([2, 2])
        var scaler = StandardScaler()
        let Xt = scaler.fitTransform(X: X)
        let Xr = scaler.inverseTransform(X: Xt)

        let original = X.asArray(Float.self)
        let recovered = Xr.asArray(Float.self)

        for (o, r) in zip(original, recovered) {
            #expect(abs(o - r) < 1e-3, "inverseTransform should recover original")
        }
    }

    @Test func transformPreservesShape() {
        let X = MLXArray([Float](repeating: 1.0, count: 12)).reshaped([4, 3])
        var scaler = StandardScaler()
        scaler.fit(X: X)
        let Xt = scaler.transform(X: X)
        #expect(Xt.shape == [4, 3])
    }
}

// MARK: - Accuracy Tests

@Suite("Accuracy", CPUDeviceTrait())
struct AccuracyTests {

    @Test func perfectPredictions() {
        let y = MLXArray([0, 1, 0, 1] as [Float]).reshaped([4, 1])
        let score = Accuracy().score(y, y)
        #expect(abs(score - 1.0) < 1e-5)
    }

    @Test func allWrong() {
        let yTrue = MLXArray([0, 0, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 1, 1, 1] as [Float]).reshaped([4, 1])
        let score = Accuracy().score(yTrue, yPred)
        #expect(abs(score) < 1e-5)
    }

    @Test func halfCorrect() {
        let yTrue = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 0, 1, 0] as [Float]).reshaped([4, 1])
        let score = Accuracy().score(yTrue, yPred)
        #expect(abs(score - 0.5) < 1e-5)
    }
}

// MARK: - ConfusionMatrix Tests

@Suite("ConfusionMatrix", CPUDeviceTrait())
struct ConfusionMatrixTests {

    @Test func perfectBinaryClassification() {
        let yTrue = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let cm = ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(cm.FP == 0)
        #expect(cm.FN == 0)
        #expect(abs(cm.accuracy - 1.0) < 1e-5)
    }

    @Test func knownConfusionValues() {
        let yTrue = MLXArray([1, 1, 0, 0, 1, 0] as [Float]).reshaped([6, 1])
        let yPred = MLXArray([1, 0, 0, 1, 1, 0] as [Float]).reshaped([6, 1])
        let cm = ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(cm.FP == 1)
        #expect(cm.FN == 1)
        #expect(abs(cm.precision - 2.0 / 3.0) < 1e-5)
        #expect(abs(cm.recall - 2.0 / 3.0) < 1e-5)
    }

    @Test func handlesSignedLabels() {
        let yTrue = MLXArray([-1, 1, -1, 1] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([-1, 1, -1, 1] as [Float]).reshaped([4, 1])
        let cm = ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(abs(cm.accuracy - 1.0) < 1e-5)
    }
}

// MARK: - BinaryCrossEntropy Tests

@Suite("BinaryCrossEntropy", CPUDeviceTrait())
struct BinaryCrossEntropyTests {

    @Test func perfectLogitsGiveLowLoss() {
        let bce = BinaryCrossEntropy()
        let logits = MLXArray([10.0, -10.0, 10.0] as [Float]).reshaped([3, 1])
        let target = MLXArray([1.0, 0.0, 1.0] as [Float]).reshaped([3, 1])
        let loss = bce.withLogits(logits: logits, target: target).item(Float.self)
        #expect(loss < 0.01)
    }

    @Test func wrongLogitsGiveHighLoss() {
        let bce = BinaryCrossEntropy()
        let logits = MLXArray([-10.0, 10.0] as [Float]).reshaped([2, 1])
        let target = MLXArray([1.0, 0.0] as [Float]).reshaped([2, 1])
        let loss = bce.withLogits(logits: logits, target: target).item(Float.self)
        #expect(loss > 5.0)
    }

    @Test func probabilityLossMatchesExpected() {
        let bce = BinaryCrossEntropy()
        let probs = MLXArray([0.99, 0.01] as [Float]).reshaped([2, 1])
        let target = MLXArray([1.0, 0.0] as [Float]).reshaped([2, 1])
        let loss = bce.withProbabilities(probs: probs, target: target).item(Float.self)
        #expect(loss < 0.02)
    }
}

// MARK: - Pipeline Tests

@Suite("Pipeline", CPUDeviceTrait())
struct PipelineTests {

    @Test func pipelineWithScalerAndModel() {
        let (X, y) = makeLinearData()
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var pipeline = Pipeline(steps: [
            .transformer(StandardScaler()),
            .model(KNN(k: 3))
        ])

        pipeline.fit(X: xTrain, y: yTrain)
        let preds = pipeline.predict(X: xTest)

        #expect(preds.shape[0] == xTest.shape[0])
        #expect(preds.shape[1] == 1)
    }
}

// MARK: - Model Smoke Tests

@Suite("LogisticRegression", CPUDeviceTrait())
struct LogisticRegressionTests {

    @Test func learnsLinearlySeparableData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = LogisticRegression(inputSize: 2, epochs: 40, learningRate: 0.1)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("SVM", CPUDeviceTrait())
struct SVMTests {

    @Test func learnsWithSignedLabels() {
        let (X, y) = makeLinearData(n: 40)
        let ySigned = 2 * y - 1  // {0,1} → {-1,+1}
        let (xTrain, yTrain, xTest, yTest) = splitData(X, ySigned)

        var model = SVM(inputSize: 2, epochs: 40, learningRate: 0.1)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let yTrue01 = MLX.where(yTest .> 0, MLXArray(1), MLXArray(0))
        let yPred01 = MLX.where(preds .> 0, MLXArray(1), MLXArray(0))
        let acc = Accuracy().score(yTrue01, yPred01)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("KNN", CPUDeviceTrait())
struct KNNTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = KNN(k: 3)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("GaussianNaiveBayes", CPUDeviceTrait())
struct GaussianNaiveBayesTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = GaussianNaiveBayes()
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("LDA", CPUDeviceTrait())
struct LDATests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = LDA()
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("QDA", CPUDeviceTrait())
struct QDATests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = QDA()
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("DecisionTree", CPUDeviceTrait())
struct DecisionTreeTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = DecisionTree(maxDepth: 3)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("RandomForest", CPUDeviceTrait())
struct RandomForestTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = RandomForest(nTrees: 3, maxDepth: 3)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("ExtraTrees", CPUDeviceTrait())
struct ExtraTreesTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = ExtraTrees(nTrees: 3, maxDepth: 3)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("GradientBoosting", CPUDeviceTrait())
struct GradientBoostingTests {

    @Test func learnsSimpleData() {
        let (X, y) = makeLinearData(n: 100)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = GradientBoosting(nEstimators: 10, learningRate: 0.1)
        model.fit(X: xTrain, y: yTrain)
        let preds = model.predict(X: xTest)

        let acc = Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

// MARK: - Determinism Tests

@Suite("Determinism", CPUDeviceTrait())
struct DeterminismTests {

    @Test func trainTestSplitSameSeedProducesSameIndices() {
        let (X, y) = makeLinearData(n: 40)

        let first = trainTestSplit(X: X, y: y, testSize: 0.25, randomState: 123, stratify: true)
        let second = trainTestSplit(X: X, y: y, testSize: 0.25, randomState: 123, stratify: true)

        #expect(first.trainIndices == second.trainIndices)
        #expect(first.testIndices == second.testIndices)
    }

    @Test func randomForestSameSeedProducesSamePredictions() {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var first = RandomForest(nTrees: 5, maxDepth: 3, randomState: 123)
        var second = RandomForest(nTrees: 5, maxDepth: 3, randomState: 123)

        first.fit(X: xTrain, y: yTrain)
        second.fit(X: xTrain, y: yTrain)

        #expect(floatValues(first.predict(X: xTest)) == floatValues(second.predict(X: xTest)))
    }

    @Test func extraTreesSameSeedProducesSamePredictions() {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var first = ExtraTrees(nTrees: 5, maxDepth: 3, randomState: 123)
        var second = ExtraTrees(nTrees: 5, maxDepth: 3, randomState: 123)

        first.fit(X: xTrain, y: yTrain)
        second.fit(X: xTrain, y: yTrain)

        #expect(floatValues(first.predict(X: xTest)) == floatValues(second.predict(X: xTest)))
    }
}
