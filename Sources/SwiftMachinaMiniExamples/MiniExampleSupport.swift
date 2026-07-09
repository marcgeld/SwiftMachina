import Foundation
import MLX
import SwiftMachina

// Shared plumbing for the one-file-per-model mini examples: a tiny synthetic
// two-cluster dataset and a fit → save JSON → load → verify round trip.
// Each estimator lives in its own <Model>MiniExample.swift file.

func makeTinyBinaryData(signedLabels: Bool = false) -> (
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

func runMiniExample<M: Model & FittedStatePersistable>(
    name: String,
    model: M,
    signedLabels: Bool = false,
    outputDirectory: URL,
    minimumAccuracy: Float = 1.0
) throws {
    let data = makeTinyBinaryData(signedLabels: signedLabels)

    var fitted = model
    try fitted.fit(X: data.xTrain, y: data.yTrain)

    let trainAccuracy = try accuracy(of: fitted, X: data.xTrain, y: data.yTrain)
    let testAccuracy = try accuracy(of: fitted, X: data.xTest, y: data.yTest)

    let fileURL = outputDirectory.appendingPathComponent("\(name).fitted-state.json")
    try fitted.saveFittedState(to: fileURL)

    let loaded = try M.loadFittedState(from: fileURL)
    let loadedAccuracy = try accuracy(of: loaded, X: data.xTest, y: data.yTest)

    guard loadedAccuracy >= minimumAccuracy else {
        throw SwiftMachinaError.invalidParameter(
            "\(name) loaded accuracy \(loadedAccuracy) below \(minimumAccuracy)"
        )
    }

    print("""
    \(name)
      train accuracy:  \(String(format: "%.2f", trainAccuracy))
      test accuracy:   \(String(format: "%.2f", testAccuracy))
      loaded accuracy: \(String(format: "%.2f", loadedAccuracy))
      saved: \(fileURL.path)
    """)
}

private func accuracy(of model: some Predictor, X: MLXArray, y: MLXArray) throws -> Float {
    let predictions = try model.predict(X: X).asType(.float32)
    return try Accuracy().score(y, predictions)
}
