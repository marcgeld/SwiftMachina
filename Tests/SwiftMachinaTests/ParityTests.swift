import Testing
import Foundation
import MLX
import TabularData
@testable import SwiftMachina

// MARK: - Cache JSON Schema

private struct CachePayload: Decodable {
    let schemaVersion: String
    let run: CacheRun
    let data: CacheData
    let metrics: CacheMetrics
    let confusionMatrix: CacheConfusion

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case run, data, metrics
        case confusionMatrix = "confusion_matrix"
    }
}

private struct CacheRun: Decodable {
    let algorithm: String
    let nSamples: Int
    let nFeatures: Int
    let seed: Int

    enum CodingKeys: String, CodingKey {
        case algorithm
        case nSamples = "n_samples"
        case nFeatures = "n_features"
        case seed
    }
}

private struct CacheData: Decodable {
    let yTrue: [Int]
    let yPred: [Int]

    enum CodingKeys: String, CodingKey {
        case yTrue = "y_true"
        case yPred = "y_pred"
    }
}

private struct CacheMetrics: Decodable {
    let accuracy: Double
    let precision: Double
    let recall: Double
    let f1: Double
    let balancedAccuracy: Double
    let specificity: Double
    let mcc: Double

    enum CodingKeys: String, CodingKey {
        case accuracy, precision, recall, f1
        case balancedAccuracy = "balanced_accuracy"
        case specificity, mcc
    }
}

private struct CacheConfusion: Decodable {
    let tp: Int
    let tn: Int
    let fp: Int
    let fn: Int
}

// MARK: - Project Paths

private let projectRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

private let cacheDir = projectRoot.appendingPathComponent("Benchtest/cache")
private let csvPath = projectRoot.appendingPathComponent("Sources/SwiftMachinaExample/Resources/breast_cancer.csv")

// MARK: - Data Loading

private func loadCache(algorithm: String) throws -> CachePayload {
    let url = cacheDir.appendingPathComponent("\(algorithm).json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(CachePayload.self, from: data)
}

private func loadBreastCancerData() throws -> (X: MLXArray, y: MLXArray) {
    let df = try DataFrame(contentsOfCSVFile: csvPath)

    // Match sklearn: malignant=0, benign=1
    let labels: [Float] = df["diagnosis"].map {
        ($0 as! String == "M") ? 0.0 : 1.0
    }

    let featureColumns = df.columns.filter { $0.name != "id" && $0.name != "diagnosis" }

    var features: [Float] = []
    features.reserveCapacity(df.rows.count * featureColumns.count)

    for row in df.rows {
        for col in featureColumns {
            features.append(Float(row[col.name] as! Double))
        }
    }

    let X = MLXArray(features).reshaped([df.rows.count, featureColumns.count])
    let y = MLXArray(labels).reshaped([df.rows.count, 1])
    return (X, y)
}

// MARK: - Model Registry
// Uses closures with concrete types to avoid existential mutation issues with `any Model`.

private typealias TrainPredict = (_ Xtrain: MLXArray, _ ytrain: MLXArray, _ Xtest: MLXArray) -> MLXArray

private func makeTrainPredict(for algorithm: String, nFeatures: Int, seed: Int) -> TrainPredict? {
    switch algorithm {
    case "decision_tree":
        return { Xtrain, ytrain, Xtest in
            var m = DecisionTree(maxDepth: 5, minSamplesSplit: 2, minSamplesLeaf: 1, minImpurityDecrease: 0.0)
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "gradient_boosting":
        return { Xtrain, ytrain, Xtest in
            var m = GradientBoosting(nEstimators: 100, learningRate: 0.1)
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "random_forest":
        return { Xtrain, ytrain, Xtest in
            var m = RandomForest(nTrees: 100, maxDepth: 5, randomState: UInt64(seed))
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "extra_trees":
        return { Xtrain, ytrain, Xtest in
            var m = ExtraTrees(nTrees: 100, maxDepth: 100, randomState: UInt64(seed))
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "knn":
        return { Xtrain, ytrain, Xtest in
            var m = KNN(k: 5)
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "lda":
        return { Xtrain, ytrain, Xtest in
            var m = LDA()
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "qda":
        return { Xtrain, ytrain, Xtest in
            var m = QDA(regParam: 0.01)
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "naive_bayes":
        return { Xtrain, ytrain, Xtest in
            var m = GaussianNaiveBayes()
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "logistic_regression":
        return { Xtrain, ytrain, Xtest in
            var m = LogisticRegression(inputSize: nFeatures, epochs: 1000, learningRate: 0.01)
            m.fit(X: Xtrain, y: ytrain)
            return m.predict(X: Xtest)
        }
    case "svm_linear":
        return { Xtrain, ytrain, Xtest in
            var m = SVM(inputSize: nFeatures, epochs: 1000, learningRate: 0.01)
            m.fit(X: Xtrain, y: 2 * ytrain - 1)
            let raw = m.predict(X: Xtest)
            return `where`(raw .> 0, MLXArray(1), MLXArray(0))
        }
    default:
        return nil
    }
}

// MARK: - Tolerances

private enum Tol {
    static let accuracy = 0.02
    static let precision = 0.02
    static let recall = 0.02
    static let f1 = 0.02
    static let balancedAccuracy = 0.02
    static let specificity = 0.02
    static let mcc = 0.03
}

// MARK: - Parity Tests

@Suite("Parity", CPUDeviceTrait())
struct ParityTests {

    @Test("Decision Tree parity")
    func decisionTree() throws { try runParity(algorithm: "decision_tree") }

    @Test("Gradient Boosting parity")
    func gradientBoosting() throws { try runParity(algorithm: "gradient_boosting") }

    @Test("Random Forest parity")
    func randomForest() throws { try runParity(algorithm: "random_forest") }

    @Test("Extra Trees parity")
    func extraTrees() throws { try runParity(algorithm: "extra_trees") }

    @Test("KNN parity")
    func knn() throws { try runParity(algorithm: "knn") }

    @Test("LDA parity")
    func lda() throws { try runParity(algorithm: "lda") }

    @Test("QDA parity")
    func qda() throws { try runParity(algorithm: "qda") }

    @Test("Naive Bayes parity")
    func naiveBayes() throws { try runParity(algorithm: "naive_bayes") }

    @Test("Logistic Regression parity")
    func logisticRegression() throws { try runParity(algorithm: "logistic_regression") }

    @Test("SVM Linear parity")
    func svmLinear() throws { try runParity(algorithm: "svm_linear") }

    private func runParity(algorithm: String) throws {
        let cache: CachePayload
        do {
            cache = try loadCache(algorithm: algorithm)
        } catch {
            print("⏭ \(algorithm): cache not found, skipping")
            return
        }

        let (X, y) = try loadBreastCancerData()

        guard let trainPredict = makeTrainPredict(
            for: algorithm,
            nFeatures: X.shape[1],
            seed: cache.run.seed
        ) else {
            print("⏭ \(algorithm): no Swift implementation, skipping")
            return
        }

        let split = trainTestSplit(
            X: X, y: y,
            testSize: 0.2,
            randomState: UInt64(cache.run.seed),
            stratify: true
        )

        let preds = trainPredict(split.Xtrain, split.ytrain, split.Xtest)
        let cm = ConfusionMatrix().compute(split.ytest, preds)

        let py = cache.metrics
        let checks: [(String, Double, Double, Double)] = [
            ("accuracy",          Double(cm.accuracy),         py.accuracy,         Tol.accuracy),
            ("precision",         Double(cm.precision),        py.precision,        Tol.precision),
            ("recall",            Double(cm.recall),           py.recall,           Tol.recall),
            ("f1",                Double(cm.f1),               py.f1,               Tol.f1),
            ("balanced_accuracy", Double(cm.balancedAccuracy), py.balancedAccuracy, Tol.balancedAccuracy),
            ("specificity",       Double(cm.specificity),      py.specificity,      Tol.specificity),
            ("mcc",               Double(cm.mcc),              py.mcc,              Tol.mcc),
        ]

        print("\n\(algorithm):")
        let hdr = "metric".padding(toLength: 20, withPad: " ", startingAt: 0)
        print("  \(hdr)    swift    python       Δ")
        print("  " + String(repeating: "─", count: 56))

        for (name, swift, python, tol) in checks {
            let delta = abs(swift - python)
            let ok = delta <= tol
            let mark = ok ? "✓" : "✗"
            let pad = name.padding(toLength: 20, withPad: " ", startingAt: 0)
            print("  \(pad)  \(String(format: "%7.4f", swift))  \(String(format: "%8.4f", python))  \(String(format: "%7.4f", delta))  \(mark)")

            #expect(
                delta <= tol,
                "\(algorithm)/\(name): swift=\(String(format: "%.4f", swift)) python=\(String(format: "%.4f", python)) Δ=\(String(format: "%.4f", delta)) exceeds ±\(tol)"
            )
        }

        print("  confusion: TP=\(cm.TP) TN=\(cm.TN) FP=\(cm.FP) FN=\(cm.FN) (swift)")
        print("             TP=\(cache.confusionMatrix.tp) TN=\(cache.confusionMatrix.tn) FP=\(cache.confusionMatrix.fp) FN=\(cache.confusionMatrix.fn) (python)")
    }
}
