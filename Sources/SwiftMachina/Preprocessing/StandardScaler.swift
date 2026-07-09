//
//  StandardScaler.swift
//  SwiftMachina
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
    public mutating func fit(X: MLXArray) throws {

        try require(X.shape.count == 2, .invalidShape("X must be 2D [N, features]"))

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
    public func transform(X: MLXArray) throws -> MLXArray {

        guard let mean, let std else {
            throw SwiftMachinaError.notFitted("StandardScaler not fitted. Call fit() first.")
        }

        try require(X.shape.count == 2, .invalidShape("X must be 2D [N, features]"))
        try require(X.shape[1] == mean.shape[0], .invalidShape("X must have the same number of features as fitted data"))
        return (X - mean) / std
    }

    // MARK: - Fit + Transform
    public mutating func fitTransform(X: MLXArray) throws -> MLXArray {
        try self.fit(X: X)
        return try self.transform(X: X)
    }

    // MARK: - Inverse Transform (valfri men bra att ha)
    public func inverseTransform(X: MLXArray) throws -> MLXArray {

        guard let mean, let std else {
            throw SwiftMachinaError.notFitted("StandardScaler not fitted.")
        }

        try require(X.shape.count == 2, .invalidShape("X must be 2D [N, features]"))
        try require(X.shape[1] == mean.shape[0], .invalidShape("X must have the same number of features as fitted data"))
        return X * std + mean
    }

    // MARK: - Debug / inspection
    public func parameters() throws -> (mean: MLXArray, std: MLXArray) {
        guard let mean, let std else {
            throw SwiftMachinaError.notFitted("StandardScaler not fitted.")
        }
        return (mean, std)
    }
}
