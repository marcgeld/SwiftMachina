//
//  Types.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Task Types

/// Describes the type of ML problem.
public enum MLTask {
    case classification
    case regression
}

// MARK: - Dataset Split

/// Simple container for train/test splits.
public struct DatasetSplit {
    public let XTrain: MLXArray
    public let XTest: MLXArray
    public let yTrain: MLXArray
    public let yTest: MLXArray

    public init(
        XTrain: MLXArray,
        XTest: MLXArray,
        yTrain: MLXArray,
        yTest: MLXArray
    ) {
        self.XTrain = XTrain
        self.XTest = XTest
        self.yTrain = yTrain
        self.yTest = yTest
    }
}

// MARK: - Prediction Result

/// Standard output from a model prediction.
public struct PredictionResult {
    public let predictions: MLXArray
    public let probabilities: MLXArray?

    public init(
        predictions: MLXArray,
        probabilities: MLXArray? = nil
    ) {
        self.predictions = predictions
        self.probabilities = probabilities
    }
}

// MARK: - Model Score

/// Generic evaluation result.
public struct ModelScore {
    public let accuracy: Float?
    public let precision: Float?
    public let recall: Float?
    public let f1: Float?

    public init(
        accuracy: Float? = nil,
        precision: Float? = nil,
        recall: Float? = nil,
        f1: Float? = nil
    ) {
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
    }
}

// MARK: - Confusion Matrix Result
/// Structured confusion matrix output.
public struct ConfusionMatrixResult {
    public let TP: Int
    public let TN: Int
    public let FP: Int
    public let FN: Int

    public init(TP: Int, TN: Int, FP: Int, FN: Int) {
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
    }
}

// MARK: - Boolean mask → Int32 indices
// MLX Gather does not support boolean indexing on all array types.
// This converts a boolean mask to int32 indices for safe fancy indexing.
public func maskToIndices(_ mask: MLXArray) -> MLXArray {
    let flat = mask.flattened().asType(.uint8)
    let vals: [UInt8] = flat.asArray(UInt8.self)
    var indices: [Int32] = []
    indices.reserveCapacity(vals.count)
    for (i, v) in vals.enumerated() {
        if v != 0 { indices.append(Int32(i)) }
    }
    return MLXArray(indices)
}

// MARK: - Hyperparameter (future use)
/// Generic hyperparameter container (useful for GridSearch later).
public struct Hyperparameter<T> {
    public let name: String
    public let values: [T]

    public init(name: String, values: [T]) {
        self.name = name
        self.values = values
    }
}
