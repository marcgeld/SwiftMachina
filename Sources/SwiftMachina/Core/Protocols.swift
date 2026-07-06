//
//  Protocols.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Base ML Protocols (sklearn-like)
//
// These protocols define the core API contracts for the SwiftMachina framework.
// Inspired by scikit-learn but adapted to Swift value semantics.

// MARK: - Estimator
/// Anything that can be trained on labeled data.
public protocol Estimator {
    /// Fit the model using training data.
    mutating func fit(X: MLXArray, y: MLXArray)
}

// MARK: - Predictor
/// Anything that can produce predictions.
public protocol Predictor {
    /// Predict class labels or values.
    func predict(X: MLXArray) -> MLXArray
}

// MARK: - Model
/// Combined type for most ML models.
public typealias Model = Estimator & Predictor

// MARK: - Classifier

/// Marker protocol for classification models.
public protocol Classifier: Model {}

// MARK: - Regressor (future use)
/// Marker protocol for regression models.
public protocol Regressor: Model {}

// MARK: - Transformer
/// Data preprocessing step (e.g. StandardScaler).
public protocol Transformer {
    /// Learn parameters from data (e.g. mean/std).
    mutating func fit(X: MLXArray)

    /// Apply transformation.
    func transform(X: MLXArray) -> MLXArray

    /// Convenience method (default implementation provided below).
    mutating func fitTransform(X: MLXArray) -> MLXArray
}

// Default implementation
public extension Transformer {
    mutating func fitTransform(X: MLXArray) -> MLXArray {
        self.fit(X: X)
        return self.transform(X: X)
    }
}
