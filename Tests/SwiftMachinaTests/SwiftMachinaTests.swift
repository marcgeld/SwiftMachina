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

private func makeTwoClusterData(
    leftLabel: Float,
    rightLabel: Float,
    nPerClass: Int = 20
) -> (X: MLXArray, y: MLXArray) {
    var features: [Float] = []
    var labels: [Float] = []

    for i in 0..<nPerClass {
        let offset = Float(i) * 0.01
        features.append(contentsOf: [-3.0 + offset, -3.0 - offset])
        labels.append(leftLabel)
    }
    for i in 0..<nPerClass {
        let offset = Float(i) * 0.01
        features.append(contentsOf: [3.0 + offset, 3.0 - offset])
        labels.append(rightLabel)
    }

    let X = MLXArray(features).reshaped([nPerClass * 2, 2])
    let y = MLXArray(labels).reshaped([nPerClass * 2, 1])
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

    @Test func fitTransformProducesZeroMeanUnitVariance() throws {
        let X = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as [Float]).reshaped([3, 2])
        var scaler = StandardScaler()
        let Xt = try scaler.fitTransform(X: X)

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

    @Test func inverseTransformRecoversOriginal() throws {
        let X = MLXArray([10.0, 20.0, 30.0, 40.0] as [Float]).reshaped([2, 2])
        var scaler = StandardScaler()
        let Xt = try scaler.fitTransform(X: X)
        let Xr = try scaler.inverseTransform(X: Xt)

        let original = X.asArray(Float.self)
        let recovered = Xr.asArray(Float.self)

        for (o, r) in zip(original, recovered) {
            #expect(abs(o - r) < 1e-3, "inverseTransform should recover original")
        }
    }

    @Test func transformPreservesShape() throws {
        let X = MLXArray([Float](repeating: 1.0, count: 12)).reshaped([4, 3])
        var scaler = StandardScaler()
        try scaler.fit(X: X)
        let Xt = try scaler.transform(X: X)
        #expect(Xt.shape == [4, 3])
    }
}

// MARK: - Accuracy Tests

@Suite("Accuracy", CPUDeviceTrait())
struct AccuracyTests {

    @Test func perfectPredictions() throws {
        let y = MLXArray([0, 1, 0, 1] as [Float]).reshaped([4, 1])
        let score = try Accuracy().score(y, y)
        #expect(abs(score - 1.0) < 1e-5)
    }

    @Test func allWrong() throws {
        let yTrue = MLXArray([0, 0, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 1, 1, 1] as [Float]).reshaped([4, 1])
        let score = try Accuracy().score(yTrue, yPred)
        #expect(abs(score) < 1e-5)
    }

    @Test func halfCorrect() throws {
        let yTrue = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 0, 1, 0] as [Float]).reshaped([4, 1])
        let score = try Accuracy().score(yTrue, yPred)
        #expect(abs(score - 0.5) < 1e-5)
    }
}

// MARK: - ConfusionMatrix Tests

@Suite("ConfusionMatrix", CPUDeviceTrait())
struct ConfusionMatrixTests {

    @Test func perfectBinaryClassification() throws {
        let yTrue = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([1, 1, 0, 0] as [Float]).reshaped([4, 1])
        let cm = try ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(cm.FP == 0)
        #expect(cm.FN == 0)
        #expect(abs(cm.accuracy - 1.0) < 1e-5)
    }

    @Test func knownConfusionValues() throws {
        let yTrue = MLXArray([1, 1, 0, 0, 1, 0] as [Float]).reshaped([6, 1])
        let yPred = MLXArray([1, 0, 0, 1, 1, 0] as [Float]).reshaped([6, 1])
        let cm = try ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(cm.FP == 1)
        #expect(cm.FN == 1)
        #expect(abs(cm.precision - 2.0 / 3.0) < 1e-5)
        #expect(abs(cm.recall - 2.0 / 3.0) < 1e-5)
    }

    @Test func handlesSignedLabels() throws {
        let yTrue = MLXArray([-1, 1, -1, 1] as [Float]).reshaped([4, 1])
        let yPred = MLXArray([-1, 1, -1, 1] as [Float]).reshaped([4, 1])
        let cm = try ConfusionMatrix().compute(yTrue, yPred)
        #expect(cm.TP == 2)
        #expect(cm.TN == 2)
        #expect(abs(cm.accuracy - 1.0) < 1e-5)
    }
}

// MARK: - BinaryCrossEntropy Tests

@Suite("BinaryCrossEntropy", CPUDeviceTrait())
struct BinaryCrossEntropyTests {

    @Test func perfectLogitsGiveLowLoss() throws {
        let bce = BinaryCrossEntropy()
        let logits = MLXArray([10.0, -10.0, 10.0] as [Float]).reshaped([3, 1])
        let target = MLXArray([1.0, 0.0, 1.0] as [Float]).reshaped([3, 1])
        let loss = try bce.withLogits(logits: logits, target: target).item(Float.self)
        #expect(loss < 0.01)
    }

    @Test func wrongLogitsGiveHighLoss() throws {
        let bce = BinaryCrossEntropy()
        let logits = MLXArray([-10.0, 10.0] as [Float]).reshaped([2, 1])
        let target = MLXArray([1.0, 0.0] as [Float]).reshaped([2, 1])
        let loss = try bce.withLogits(logits: logits, target: target).item(Float.self)
        #expect(loss > 5.0)
    }

    @Test func probabilityLossMatchesExpected() throws {
        let bce = BinaryCrossEntropy()
        let probs = MLXArray([0.99, 0.01] as [Float]).reshaped([2, 1])
        let target = MLXArray([1.0, 0.0] as [Float]).reshaped([2, 1])
        let loss = try bce.withProbabilities(probs: probs, target: target).item(Float.self)
        #expect(loss < 0.02)
    }
}

// MARK: - Pipeline Tests

@Suite("Pipeline", CPUDeviceTrait())
struct PipelineTests {

    @Test func pipelineWithScalerAndModel() throws {
        let (X, y) = makeLinearData()
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var pipeline = try Pipeline(steps: [
            .transformer(StandardScaler()),
            .model(try KNN(k: 3))
        ])

        try pipeline.fit(X: xTrain, y: yTrain)
        let preds = try pipeline.predict(X: xTest)

        #expect(preds.shape[0] == xTest.shape[0])
        #expect(preds.shape[1] == 1)
    }
}

// MARK: - Throwing API Tests

@Suite("Throwing API", CPUDeviceTrait())
struct ThrowingAPITests {

    @Test func pipelineRejectsMissingFinalModel() {
        var didThrow = false

        do {
            _ = try Pipeline(steps: [
                .transformer(StandardScaler())
            ])
        } catch {
            didThrow = true
            #expect((error as? SwiftMachinaError) == .invalidPipeline("Pipeline must end with a model"))
        }

        #expect(didThrow)
    }

    @Test func trainTestSplitRejectsInvalidTestSize() {
        let (X, y) = makeLinearData(n: 10)
        var didThrow = false

        do {
            _ = try trainTestSplit(X: X, y: y, testSize: 1.0)
        } catch {
            didThrow = true
            #expect((error as? SwiftMachinaError) == .invalidParameter("testSize must be in (0, 1)"))
        }

        #expect(didThrow)
    }

    @Test func predictBeforeFitThrows() throws {
        let (X, _) = makeLinearData(n: 10)
        let model = try KNN(k: 3)
        var didThrow = false

        do {
            _ = try model.predict(X: X)
        } catch {
            didThrow = true
            #expect((error as? SwiftMachinaError) == .notFitted("KNN not fitted"))
        }

        #expect(didThrow)
    }
}

// MARK: - Model Smoke Tests

@Suite("LogisticRegression", CPUDeviceTrait())
struct LogisticRegressionTests {

    @Test func learnsLinearlySeparableData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try LogisticRegression(inputSize: 2, epochs: 40, learningRate: 0.1)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("SVM", CPUDeviceTrait())
struct SVMTests {

    @Test func learnsWithSignedLabels() throws {
        let (X, y) = makeLinearData(n: 40)
        let ySigned = 2 * y - 1  // {0,1} → {-1,+1}
        let (xTrain, yTrain, xTest, yTest) = splitData(X, ySigned)

        var model = try SVM(inputSize: 2, epochs: 40, learningRate: 0.1)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let yTrue01 = MLX.where(yTest .> 0, MLXArray(1), MLXArray(0))
        let yPred01 = MLX.where(preds .> 0, MLXArray(1), MLXArray(0))
        let acc = try Accuracy().score(yTrue01, yPred01)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("KNN", CPUDeviceTrait())
struct KNNTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try KNN(k: 3)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func preservesSignedLabels() throws {
        let (X, y) = makeTwoClusterData(leftLabel: -1, rightLabel: 1)

        var model = try KNN(k: 3)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        #expect(try Accuracy().score(y, preds) == 1.0)
        #expect(Set(floatValues(preds)) == Set<Float>([-1.0, 1.0]))
    }
}

@Suite("GaussianNaiveBayes", CPUDeviceTrait())
struct GaussianNaiveBayesTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = GaussianNaiveBayes()
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func refitReplacesLearnedParameters() throws {
        let first = makeTwoClusterData(leftLabel: 0, rightLabel: 1)
        let second = makeTwoClusterData(leftLabel: 3, rightLabel: 2)

        var model = GaussianNaiveBayes()
        try model.fit(X: first.X, y: first.y)
        try model.fit(X: second.X, y: second.y)

        #expect(try Accuracy().score(second.y, try model.predict(X: second.X)) == 1.0)
    }
}

@Suite("LDA", CPUDeviceTrait())
struct LDATests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = LDA()
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func refitReplacesLearnedParameters() throws {
        let first = makeTwoClusterData(leftLabel: 0, rightLabel: 1)
        let second = makeTwoClusterData(leftLabel: 3, rightLabel: 2)

        var model = LDA()
        try model.fit(X: first.X, y: first.y)
        try model.fit(X: second.X, y: second.y)

        #expect(try Accuracy().score(second.y, try model.predict(X: second.X)) == 1.0)
    }
}

@Suite("QDA", CPUDeviceTrait())
struct QDATests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try QDA()
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func refitReplacesLearnedParameters() throws {
        let first = makeTwoClusterData(leftLabel: 0, rightLabel: 1)
        let second = makeTwoClusterData(leftLabel: 3, rightLabel: 2)

        var model = try QDA()
        try model.fit(X: first.X, y: first.y)
        try model.fit(X: second.X, y: second.y)

        #expect(try Accuracy().score(second.y, try model.predict(X: second.X)) == 1.0)
    }
}

@Suite("DecisionTree", CPUDeviceTrait())
struct DecisionTreeTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try DecisionTree(maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }
}

@Suite("RandomForest", CPUDeviceTrait())
struct RandomForestTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try RandomForest(nTrees: 3, maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func preservesSignedLabels() throws {
        let (X, y) = makeTwoClusterData(leftLabel: -1, rightLabel: 1)

        var model = try RandomForest(nTrees: 9, maxDepth: 2, randomState: 123)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        #expect(try Accuracy().score(y, preds) >= 0.95)
        #expect(Set(floatValues(preds)).isSubset(of: Set<Float>([-1.0, 1.0])))
    }
}

@Suite("ExtraTrees", CPUDeviceTrait())
struct ExtraTreesTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 40)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try ExtraTrees(nTrees: 3, maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func preservesSignedLabels() throws {
        let (X, y) = makeTwoClusterData(leftLabel: -1, rightLabel: 1)

        var model = try ExtraTrees(nTrees: 9, maxDepth: 3, randomState: 123)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        #expect(try Accuracy().score(y, preds) >= 0.95)
        #expect(Set(floatValues(preds)).isSubset(of: Set<Float>([-1.0, 1.0])))
    }
}

@Suite("GradientBoosting", CPUDeviceTrait())
struct GradientBoostingTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 100)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try GradientBoosting(nEstimators: 10, learningRate: 0.1)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func preservesArbitraryBinaryLabels() throws {
        let (X, y) = makeTwoClusterData(leftLabel: 3, rightLabel: 2)

        var model = try GradientBoosting(nEstimators: 20, learningRate: 0.2)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        #expect(try Accuracy().score(y, preds) >= 0.95)
        #expect(Set(floatValues(preds)) == Set<Float>([2.0, 3.0]))
    }
}

@Suite("XGBoost", CPUDeviceTrait())
struct XGBoostTests {

    @Test func learnsSimpleData() throws {
        let (X, y) = makeLinearData(n: 100)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try XGBoostClassifier(nEstimators: 20, learningRate: 0.3, maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain)
        let preds = try model.predict(X: xTest)

        let acc = try Accuracy().score(yTest, preds)
        #expect(acc >= 0.75, "Expected >=75% accuracy, got \(acc)")
    }

    @Test func preservesArbitraryBinaryLabels() throws {
        let (X, y) = makeTwoClusterData(leftLabel: 3, rightLabel: 2)

        var model = try XGBoostClassifier(nEstimators: 20, learningRate: 0.3, maxDepth: 3)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        #expect(try Accuracy().score(y, preds) >= 0.95)
        #expect(Set(floatValues(preds)) == Set<Float>([2.0, 3.0]))
    }

    @Test func predictProbaReturnsValidProbabilities() throws {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var model = try XGBoostClassifier(nEstimators: 10, learningRate: 0.3, maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain)
        let probs = try model.predictProba(X: xTest)

        #expect(probs.shape == [xTest.shape[0], 1])
        for p in floatValues(probs) {
            #expect(p >= 0 && p <= 1, "Probability out of range: \(p)")
        }
    }

    @Test func highGammaPreventsSplits() throws {
        let (X, y) = makeLinearData(n: 60)

        var model = try XGBoostClassifier(nEstimators: 5, learningRate: 0.3, maxDepth: 3, gamma: 1e9)
        try model.fit(X: X, y: y)
        let preds = try model.predict(X: X)

        // With an enormous split penalty every tree is a single leaf,
        // so all predictions collapse to one class.
        #expect(Set(floatValues(preds)).count == 1)
    }

    @Test func regularizationShrinksConfidence() throws {
        let (X, y) = makeLinearData(n: 60)

        var weak = try XGBoostClassifier(nEstimators: 10, learningRate: 0.3, maxDepth: 3, lambda: 100.0)
        var strong = try XGBoostClassifier(nEstimators: 10, learningRate: 0.3, maxDepth: 3, lambda: 0.0)
        try weak.fit(X: X, y: y)
        try strong.fit(X: X, y: y)

        // Heavier L2 keeps probabilities closer to the base rate (0.5).
        let weakSpread = floatValues(try weak.predictProba(X: X)).map { abs($0 - 0.5) }.max()!
        let strongSpread = floatValues(try strong.predictProba(X: X)).map { abs($0 - 0.5) }.max()!
        #expect(weakSpread < strongSpread)
    }

    @Test func earlyStoppingTruncatesRounds() throws {
        let (X, y) = makeLinearData(n: 100)
        let (xTrain, yTrain, xTest, yTest) = splitData(X, y)

        var model = try XGBoostClassifier(nEstimators: 200, learningRate: 0.3, maxDepth: 3)
        try model.fit(X: xTrain, y: yTrain, evalX: xTest, evalY: yTest, earlyStoppingRounds: 5)

        #expect(model.bestIteration != nil)
        #expect(model.boostedRounds < 200, "Early stopping should truncate, kept \(model.boostedRounds)")
        #expect(model.boostedRounds == model.bestIteration! + 1)
    }

    @Test func predictBeforeFitThrows() throws {
        let (X, _) = makeLinearData(n: 10)
        let model = try XGBoostClassifier()
        var didThrow = false

        do {
            _ = try model.predict(X: X)
        } catch {
            didThrow = true
            #expect((error as? SwiftMachinaError) == .notFitted("XGBoostClassifier must be fitted before prediction"))
        }

        #expect(didThrow)
    }
}

// MARK: - Determinism Tests

@Suite("Determinism", CPUDeviceTrait())
struct DeterminismTests {

    @Test func trainTestSplitSameSeedProducesSameIndices() throws {
        let (X, y) = makeLinearData(n: 40)

        let first = try trainTestSplit(X: X, y: y, testSize: 0.25, randomState: 123, stratify: true)
        let second = try trainTestSplit(X: X, y: y, testSize: 0.25, randomState: 123, stratify: true)

        #expect(first.trainIndices == second.trainIndices)
        #expect(first.testIndices == second.testIndices)
    }

    @Test func randomForestSameSeedProducesSamePredictions() throws {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var first = try RandomForest(nTrees: 5, maxDepth: 3, randomState: 123)
        var second = try RandomForest(nTrees: 5, maxDepth: 3, randomState: 123)

        try first.fit(X: xTrain, y: yTrain)
        try second.fit(X: xTrain, y: yTrain)

        #expect(floatValues(try first.predict(X: xTest)) == floatValues(try second.predict(X: xTest)))
    }

    @Test func extraTreesSameSeedProducesSamePredictions() throws {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var first = try ExtraTrees(nTrees: 5, maxDepth: 3, randomState: 123)
        var second = try ExtraTrees(nTrees: 5, maxDepth: 3, randomState: 123)

        try first.fit(X: xTrain, y: yTrain)
        try second.fit(X: xTrain, y: yTrain)

        #expect(floatValues(try first.predict(X: xTest)) == floatValues(try second.predict(X: xTest)))
    }

    @Test func xgboostSameSeedProducesSamePredictions() throws {
        let (X, y) = makeLinearData(n: 60)
        let (xTrain, yTrain, xTest, _) = splitData(X, y)

        var first = try XGBoostClassifier(
            nEstimators: 10, maxDepth: 3, subsample: 0.7, colsampleByTree: 0.5, randomState: 123
        )
        var second = try XGBoostClassifier(
            nEstimators: 10, maxDepth: 3, subsample: 0.7, colsampleByTree: 0.5, randomState: 123
        )

        try first.fit(X: xTrain, y: yTrain)
        try second.fit(X: xTrain, y: yTrain)

        #expect(floatValues(try first.predictProba(X: xTest)) == floatValues(try second.predictProba(X: xTest)))
    }
}
