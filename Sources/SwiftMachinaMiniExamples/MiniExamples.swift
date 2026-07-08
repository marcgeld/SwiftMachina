import Foundation
import MLX
import SwiftMachina

// Tiny fitted-state round trip for every estimator: train, save JSON state,
// load the fitted model, and verify predictions without refitting.

private enum ModelKind: String, Codable, CaseIterable {
    case logisticRegression
    case svm
    case knn
    case gaussianNaiveBayes
    case lda
    case qda
    case decisionTree
    case randomForest
    case extraTrees
    case gradientBoosting
    case xgBoost
}

private struct MiniScenario {
    let name: String
    let kind: ModelKind
    let model: MiniModel
    let xTrain: MLXArray
    let yTrain: MLXArray
    let xTest: MLXArray
    let yTest: MLXArray
    let minimumAccuracy: Float
}

private enum MiniModel {
    case logisticRegression(LogisticRegression)
    case svm(SVM)
    case knn(KNN)
    case gaussianNaiveBayes(GaussianNaiveBayes)
    case lda(LDA)
    case qda(QDA)
    case decisionTree(DecisionTree)
    case randomForest(RandomForest)
    case extraTrees(ExtraTrees)
    case gradientBoosting(GradientBoosting)
    case xgBoost(XGBoostClassifier)

    mutating func fit(X: MLXArray, y: MLXArray) throws {
        switch self {
        case .logisticRegression(var model):
            try model.fit(X: X, y: y)
            self = .logisticRegression(model)
        case .svm(var model):
            try model.fit(X: X, y: y)
            self = .svm(model)
        case .knn(var model):
            try model.fit(X: X, y: y)
            self = .knn(model)
        case .gaussianNaiveBayes(var model):
            try model.fit(X: X, y: y)
            self = .gaussianNaiveBayes(model)
        case .lda(var model):
            try model.fit(X: X, y: y)
            self = .lda(model)
        case .qda(var model):
            try model.fit(X: X, y: y)
            self = .qda(model)
        case .decisionTree(var model):
            try model.fit(X: X, y: y)
            self = .decisionTree(model)
        case .randomForest(var model):
            try model.fit(X: X, y: y)
            self = .randomForest(model)
        case .extraTrees(var model):
            try model.fit(X: X, y: y)
            self = .extraTrees(model)
        case .gradientBoosting(var model):
            try model.fit(X: X, y: y)
            self = .gradientBoosting(model)
        case .xgBoost(var model):
            try model.fit(X: X, y: y)
            self = .xgBoost(model)
        }
    }

    func predict(X: MLXArray) throws -> MLXArray {
        switch self {
        case .logisticRegression(let model):
            return try model.predict(X: X).asType(.float32)
        case .svm(let model):
            return try model.predict(X: X).asType(.float32)
        case .knn(let model):
            return try model.predict(X: X)
        case .gaussianNaiveBayes(let model):
            return try model.predict(X: X)
        case .lda(let model):
            return try model.predict(X: X)
        case .qda(let model):
            return try model.predict(X: X)
        case .decisionTree(let model):
            return try model.predict(X: X)
        case .randomForest(let model):
            return try model.predict(X: X)
        case .extraTrees(let model):
            return try model.predict(X: X)
        case .gradientBoosting(let model):
            return try model.predict(X: X)
        case .xgBoost(let model):
            return try model.predict(X: X)
        }
    }

    func saveFittedState(to url: URL) throws {
        switch self {
        case .logisticRegression(let model):
            try model.saveFittedState(to: url)
        case .svm(let model):
            try model.saveFittedState(to: url)
        case .knn(let model):
            try model.saveFittedState(to: url)
        case .gaussianNaiveBayes(let model):
            try model.saveFittedState(to: url)
        case .lda(let model):
            try model.saveFittedState(to: url)
        case .qda(let model):
            try model.saveFittedState(to: url)
        case .decisionTree(let model):
            try model.saveFittedState(to: url)
        case .randomForest(let model):
            try model.saveFittedState(to: url)
        case .extraTrees(let model):
            try model.saveFittedState(to: url)
        case .gradientBoosting(let model):
            try model.saveFittedState(to: url)
        case .xgBoost(let model):
            try model.saveFittedState(to: url)
        }
    }

    static func load(kind: ModelKind, from url: URL) throws -> MiniModel {
        switch kind {
        case .logisticRegression:
            return .logisticRegression(try LogisticRegression.loadFittedState(from: url))
        case .svm:
            return .svm(try SVM.loadFittedState(from: url))
        case .knn:
            return .knn(try KNN.loadFittedState(from: url))
        case .gaussianNaiveBayes:
            return .gaussianNaiveBayes(try GaussianNaiveBayes.loadFittedState(from: url))
        case .lda:
            return .lda(try LDA.loadFittedState(from: url))
        case .qda:
            return .qda(try QDA.loadFittedState(from: url))
        case .decisionTree:
            return .decisionTree(try DecisionTree.loadFittedState(from: url))
        case .randomForest:
            return .randomForest(try RandomForest.loadFittedState(from: url))
        case .extraTrees:
            return .extraTrees(try ExtraTrees.loadFittedState(from: url))
        case .gradientBoosting:
            return .gradientBoosting(try GradientBoosting.loadFittedState(from: url))
        case .xgBoost:
            return .xgBoost(try XGBoostClassifier.loadFittedState(from: url))
        }
    }
}

private func makeTinyBinaryData(signedLabels: Bool = false) -> (
    xTrain: MLXArray,
    yTrain: MLXArray,
    xTest: MLXArray,
    yTest: MLXArray
) {
    let xTrain = MLXArray([
        -3.0, -3.0,
        -2.5, -3.5,
        -3.5, -2.5,
        -2.0, -2.0,
         3.0,  3.0,
         2.5,  3.5,
         3.5,  2.5,
         2.0,  2.0
    ] as [Float]).reshaped([8, 2])

    let xTest = MLXArray([
        -2.75, -2.75,
        -3.25, -2.25,
         2.75,  2.75,
         3.25,  2.25
    ] as [Float]).reshaped([4, 2])

    if signedLabels {
        return (
            xTrain,
            MLXArray([-1, -1, -1, -1, 1, 1, 1, 1] as [Float]).reshaped([8, 1]),
            xTest,
            MLXArray([-1, -1, 1, 1] as [Float]).reshaped([4, 1])
        )
    }

    return (
        xTrain,
        MLXArray([0, 0, 0, 0, 1, 1, 1, 1] as [Float]).reshaped([8, 1]),
        xTest,
        MLXArray([0, 0, 1, 1] as [Float]).reshaped([4, 1])
    )
}

private func scenarioFor(kind: ModelKind) throws -> MiniScenario {
    let signedLabels = kind == .svm
    let data = makeTinyBinaryData(signedLabels: signedLabels)

    let model: MiniModel
    switch kind {
    case .logisticRegression:
        model = .logisticRegression(try LogisticRegression(inputSize: 2, epochs: 120, learningRate: 0.1))
    case .svm:
        model = .svm(try SVM(inputSize: 2, epochs: 100, learningRate: 0.1))
    case .knn:
        model = .knn(try KNN(k: 1))
    case .gaussianNaiveBayes:
        model = .gaussianNaiveBayes(GaussianNaiveBayes())
    case .lda:
        model = .lda(LDA())
    case .qda:
        model = .qda(try QDA(regParam: 0.1))
    case .decisionTree:
        model = .decisionTree(try DecisionTree(maxDepth: 2))
    case .randomForest:
        model = .randomForest(try RandomForest(nTrees: 5, maxDepth: 2, randomState: 7))
    case .extraTrees:
        model = .extraTrees(try ExtraTrees(nTrees: 5, maxDepth: 2, randomState: 7))
    case .gradientBoosting:
        model = .gradientBoosting(try GradientBoosting(nEstimators: 8, learningRate: 0.3, maxDepth: 2))
    case .xgBoost:
        model = .xgBoost(try XGBoostClassifier(nEstimators: 8, learningRate: 0.3, maxDepth: 2, randomState: 7))
    }

    return MiniScenario(
        name: displayName(for: kind),
        kind: kind,
        model: model,
        xTrain: data.xTrain,
        yTrain: data.yTrain,
        xTest: data.xTest,
        yTest: data.yTest,
        minimumAccuracy: 1.0
    )
}

private func displayName(for kind: ModelKind) -> String {
    switch kind {
    case .logisticRegression: return "LogisticRegression"
    case .svm: return "SVM"
    case .knn: return "KNN"
    case .gaussianNaiveBayes: return "GaussianNaiveBayes"
    case .lda: return "LDA"
    case .qda: return "QDA"
    case .decisionTree: return "DecisionTree"
    case .randomForest: return "RandomForest"
    case .extraTrees: return "ExtraTrees"
    case .gradientBoosting: return "GradientBoosting"
    case .xgBoost: return "XGBoostClassifier"
    }
}

private func fileName(for kind: ModelKind) -> String {
    "\(kind.rawValue).fitted-state.json"
}

private func accuracy(model: MiniModel, X: MLXArray, y: MLXArray) throws -> Float {
    let predictions = try model.predict(X: X)
    return try Accuracy().score(y, predictions)
}

private func run(scenario: MiniScenario, outputDirectory: URL) throws {
    var model = scenario.model
    try model.fit(X: scenario.xTrain, y: scenario.yTrain)

    let trainAccuracy = try accuracy(model: model, X: scenario.xTrain, y: scenario.yTrain)
    let testAccuracy = try accuracy(model: model, X: scenario.xTest, y: scenario.yTest)

    let fileURL = outputDirectory.appendingPathComponent(fileName(for: scenario.kind))
    try model.saveFittedState(to: fileURL)

    let loadedModel = try MiniModel.load(kind: scenario.kind, from: fileURL)

    let loadedAccuracy = try accuracy(
        model: loadedModel,
        X: scenario.xTest,
        y: scenario.yTest
    )

    guard loadedAccuracy >= scenario.minimumAccuracy else {
        throw SwiftMachinaError.invalidParameter(
            "\(scenario.name) loaded accuracy \(loadedAccuracy) below \(scenario.minimumAccuracy)"
        )
    }

    print("""
    \(scenario.name)
      train accuracy:  \(String(format: "%.2f", trainAccuracy))
      test accuracy:   \(String(format: "%.2f", testAccuracy))
      loaded accuracy: \(String(format: "%.2f", loadedAccuracy))
      saved: \(fileURL.path)
    """)
}

@main
struct SwiftMachinaMiniExamples {
    static func main() throws {
        try Device.withDefaultDevice(.cpu) {
            let outputDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("MiniModelBundles", isDirectory: true)

            try FileManager.default.createDirectory(
                at: outputDirectory,
                withIntermediateDirectories: true
            )

            for kind in ModelKind.allCases {
                try run(scenario: scenarioFor(kind: kind), outputDirectory: outputDirectory)
            }

            print("\nAll fitted-state model JSON files loaded and verified.")
        }
    }
}
