//
//  Accuracy.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Accuracy Metric
// Computes classification accuracy:
//
//      accuracy = (# correct predictions) / (total samples)
//
// Works for binary and multiclass classification,
// as long as predictions and labels have matching shapes.

public struct Accuracy {

    public init() {}

    // MARK: - Score
    public func score(_ yTrue: MLXArray, _ yPred: MLXArray) -> Float {

        // Ensure shapes are compatible
        precondition(
            yTrue.shape == yPred.shape,
            "yTrue and yPred must have the same shape"
        )

        // Compare element-wise (1 if equal, 0 otherwise)
        let correct = (yTrue .== yPred).asType(.float32)

        // Sum correct predictions
        let correctCount = correct.sum()

        // Number of samples (assumes shape [N, ...])
        let total = Float(yTrue.shape[0])

        return (correctCount / total).item(Float.self)
    }
}
