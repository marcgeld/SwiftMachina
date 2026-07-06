import Foundation
import MLX
import SwiftMachina

private func makeLinearData(n: Int = 1_000) -> (X: MLXArray, y: MLXArray) {
    let half = n / 2

    var features: [Float] = []
    features.reserveCapacity(n * 2)

    var labels: [Float] = []
    labels.reserveCapacity(n)

    for index in 0..<half {
        let offset = Float(index) * 0.002
        features.append(contentsOf: [-2.0 + offset, -2.0 + offset])
        labels.append(0)
    }

    for index in 0..<half {
        let offset = Float(index) * 0.002
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

private struct BenchmarkResult {
    let name: String
    let accuracy: Double
    let trainTime: Double
    let predictTime: Double
}

private func runBenchmark(
    name: String,
    model: any Model,
    xTrain: MLXArray,
    yTrain: MLXArray,
    xTest: MLXArray,
    yTest: MLXArray
) throws -> BenchmarkResult {
    var pipeline = try Pipeline(steps: [
        .transformer(StandardScaler()),
        .model(model)
    ])

    let trainStart = CFAbsoluteTimeGetCurrent()
    try pipeline.fit(X: xTrain, y: yTrain)
    let trainTime = CFAbsoluteTimeGetCurrent() - trainStart

    let predictStart = CFAbsoluteTimeGetCurrent()
    let rawPredictions = try pipeline.predict(X: xTest)
    let predictTime = CFAbsoluteTimeGetCurrent() - predictStart

    let predictions: MLXArray
    let expected: MLXArray

    if name == "SVM" {
        predictions = MLX.where(rawPredictions .> 0, MLXArray(1), MLXArray(0))
        expected = MLX.where(yTest .> 0, MLXArray(1), MLXArray(0))
    } else {
        predictions = rawPredictions
        expected = yTest
    }

    return BenchmarkResult(
        name: name,
        accuracy: Double(try Accuracy().score(expected, predictions)),
        trainTime: trainTime,
        predictTime: predictTime
    )
}

@main
struct SwiftMachinaBenchmarks {
    static func main() throws {
        try Device.withDefaultDevice(.cpu) {
            let (X, y) = makeLinearData()
            let (xTrain, yTrain, xTest, yTest) = splitData(X, y)
            let yTrainSigned = 2 * yTrain - 1
            let yTestSigned = 2 * yTest - 1

            let runs: [(String, any Model, MLXArray, MLXArray)] = [
                ("LogisticRegression", try LogisticRegression(inputSize: 2, epochs: 300, learningRate: 0.1), yTrain, yTest),
                ("SVM", try SVM(inputSize: 2, epochs: 300, learningRate: 0.1), yTrainSigned, yTestSigned),
                ("KNN", try KNN(k: 3), yTrain, yTest),
                ("GaussianNaiveBayes", GaussianNaiveBayes(), yTrain, yTest),
                ("LDA", LDA(), yTrain, yTest),
                ("QDA", try QDA(), yTrain, yTest),
                ("DecisionTree", try DecisionTree(maxDepth: 5), yTrain, yTest),
                ("RandomForest", try RandomForest(nTrees: 10, maxDepth: 5, randomState: 42), yTrain, yTest),
                ("ExtraTrees", try ExtraTrees(nTrees: 10, maxDepth: 5, randomState: 42), yTrain, yTest),
                ("GradientBoosting", try GradientBoosting(nEstimators: 10, learningRate: 0.1), yTrain, yTest)
            ]

            print("SwiftMachina benchmarks on synthetic linear data")
            print("Model | Accuracy | Train (s) | Predict (s)")
            print("--- | ---: | ---: | ---:")

            for (name, model, benchmarkYTrain, benchmarkYTest) in runs {
                let result = try runBenchmark(
                    name: name,
                    model: model,
                    xTrain: xTrain,
                    yTrain: benchmarkYTrain,
                    xTest: xTest,
                    yTest: benchmarkYTest
                )

                print(
                    "\(result.name) | " +
                    "\(String(format: "%.4f", result.accuracy)) | " +
                    "\(String(format: "%.4f", result.trainTime)) | " +
                    "\(String(format: "%.4f", result.predictTime))"
                )
            }
        }
    }
}
