//
//  StandardScaler.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - StandardScaler
// Skalar features till mean=0, std=1 (per kolumn)

public struct StandardScaler: Transformer {

    // MARK: - Learned parameters
    private var mean: MLXArray?
    private var std: MLXArray?

    // För numerisk stabilitet
    private let epsilon: Float = 1e-8

    public init() {}

    // MARK: - Fit
    public mutating func fit(X: MLXArray) {

        precondition(X.shape.count == 2, "X must be 2D [N, features]")

        // mean över samples (axis 0)
        let mean = X.mean(axis: 0)

        // variance
        let variance = ((X - mean) * (X - mean)).mean(axis: 0)

        // std
        let std = sqrt(variance + epsilon)

        self.mean = mean
        self.std = std
    }

    // MARK: - Transform
    public func transform(X: MLXArray) -> MLXArray {

        guard let mean, let std else {
            fatalError("StandardScaler not fitted. Call fit() first.")
        }

        return (X - mean) / std
    }

    // MARK: - Fit + Transform
    public mutating func fitTransform(X: MLXArray) -> MLXArray {
        self.fit(X: X)
        return self.transform(X: X)
    }

    // MARK: - Inverse Transform (valfri men bra att ha)
    public func inverseTransform(X: MLXArray) -> MLXArray {

        guard let mean, let std else {
            fatalError("StandardScaler not fitted.")
        }

        return X * std + mean
    }

    // MARK: - Debug / inspection
    public func parameters() -> (mean: MLXArray, std: MLXArray) {
        guard let mean, let std else {
            fatalError("StandardScaler not fitted.")
        }
        return (mean, std)
    }
}
