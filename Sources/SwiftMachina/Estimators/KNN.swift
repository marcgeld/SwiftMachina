//
//  KNN.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - K-Nearest Neighbors (Vectorized)
//
// Efficient implementation using matrix operations.
// Computes full distance matrix in one pass.
//
// Works for binary classification (0/1 labels)

public struct KNN: Classifier {

    private var Xtrain: MLXArray?
    private var ytrain: MLXArray?
    private var classes: [Float] = []

    public let k: Int

    public init(k: Int = 3) throws {
        try require(k > 0, .invalidParameter("k must be greater than zero"))
        self.k = k
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))
        self.Xtrain = X
        self.ytrain = y
        self.classes = Array(Set(y.flattened().asArray(Float.self))).sorted()
    }

    // MARK: - Predict (vectorized)
    public func predict(X: MLXArray) throws -> MLXArray {

        guard let Xtrain, let ytrain else {
            throw SwiftMachinaError.notFitted("KNN not fitted")
        }

        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == Xtrain.shape[1], .invalidShape("X must have the same number of features as training data"))
        try require(k <= Xtrain.shape[0], .invalidParameter("k must be less than or equal to the number of training samples"))

        let nTest = X.shape[0]

        // MARK: - Compute squared distances (vectorized)

        // ||X||^2  → shape [nTest, 1]
        let Xnorm = (X * X).sum(axis: 1).reshaped([nTest, 1])

        // ||Xtrain||^2 → shape [1, nTrain]
        let XtrainNorm = (Xtrain * Xtrain).sum(axis: 1).reshaped([1, Xtrain.shape[0]])

        // Cross term
        let cross = matmul(X, Xtrain.T)

        // Full distance matrix: [nTest, nTrain]
        let dist = Xnorm + XtrainNorm - 2 * cross

        // MARK: - Find k nearest neighbors
        // argsort per row → indices of nearest neighbors
        let sortedIdx = argSort(dist, axis: 1)

        // Take first k columns
        let kIdx = sortedIdx[0..<nTest, 0..<k]

        // Gather labels
        let neighbors = ytrain[kIdx]

        // MARK: - Majority vote
        let neighborLabels = neighbors.asArray(Float.self)
        let preds = (0..<nTest).map { row in
            let start = row * k
            let rowLabels = Array(neighborLabels[start..<(start + k)])
            return Self.majorityLabel(rowLabels, classes: classes)
        }
        return MLXArray(preds).reshaped([nTest, 1])
    }

    private static func majorityLabel(_ labels: [Float], classes: [Float]) -> Float {
        var counts: [Float: Int] = [:]
        for label in labels {
            counts[label, default: 0] += 1
        }
        return classes.max { lhs, rhs in
            let left = counts[lhs, default: 0]
            let right = counts[rhs, default: 0]
            return left == right ? lhs > rhs : left < right
        }!
    }
}

extension KNN: FittedStatePersistable {
    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let k: Int
        public let xTrain: SwiftMachinaArray
        public let yTrain: SwiftMachinaArray
        public let classes: [Float]
    }

    public func fittedState() throws -> FittedState {
        guard let Xtrain, let ytrain else {
            throw SwiftMachinaError.notFitted("KNN must be fitted before saving")
        }

        return FittedState(
            schemaVersion: fittedStateSchemaVersion,
            modelType: "KNN",
            k: k,
            xTrain: SwiftMachinaArray(Xtrain),
            yTrain: SwiftMachinaArray(ytrain),
            classes: classes
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "KNN"
        )
        try require(fittedState.k > 0, .invalidParameter("k must be greater than zero"))

        let xTrain = try fittedState.xTrain.mlxArray()
        let yTrain = try fittedState.yTrain.mlxArray()
        try require(xTrain.shape.count == 2, .invalidShape("xTrain must be a 2D array"))
        try requireLabelVector(yTrain, rows: xTrain.shape[0], name: "yTrain")
        try require(xTrain.shape[0] > 0, .invalidShape("xTrain must contain at least one sample"))
        try require(!fittedState.classes.isEmpty, .notFitted("KNN fitted state must contain classes"))

        let labels = Set(yTrain.flattened().asArray(Float.self))
        try require(
            labels.isSubset(of: Set(fittedState.classes)),
            .invalidParameter("KNN classes must include every yTrain label")
        )

        self.k = fittedState.k
        self.Xtrain = xTrain
        self.ytrain = yTrain
        self.classes = fittedState.classes
    }
}
