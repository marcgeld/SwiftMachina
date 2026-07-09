//
//  LDA.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Linear Discriminant Analysis
//
// Assumes shared covariance across classes → linear decision boundary.

public struct LDA: Classifier {

    private var classes: [Float] = []
    private var means: [MLXArray] = []
    private var priors: [Float] = []
    private var invCov: MLXArray?

    public init() {}

    // MARK: - Fit

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()
        means = []
        priors = []
        invCov = nil

        var cov = MLXArray.zeros([X.shape[1], X.shape[1]])

        for c in classes {

            let idx = maskToIndices(yFlat .== c)
            let Xc = X[idx]

            let mean = Xc.mean(axis: 0)
            means.append(mean)

            let centered = Xc - mean
            cov += matmul(centered.T, centered)

            let prior = Float(Xc.shape[0]) / Float(X.shape[0])
            priors.append(prior)
        }

        cov /= Float(X.shape[0])
        let reg = Float(1e-6) * MLXArray.eye(X.shape[1])
        let regularized = cov + reg

        // SwiftNumerica inverts in Double precision; MLX (Float32, also on
        // the CPU stream) is the fallback for singular input.
        invCov = numericaInverse(regularized) ?? inv(regularized, stream: .cpu)
    }

    // MARK: - Predict

    public func predict(X: MLXArray) throws -> MLXArray {

        guard let invCov else {
            throw SwiftMachinaError.notFitted("LDA not fitted")
        }
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == invCov.shape[0], .invalidShape("X must have the same number of features as training data"))

        var scores: [MLXArray] = []

        for i in 0..<classes.count {

            let mean = means[i]

            let term1 = matmul(matmul(X, invCov), mean)
            let term2 = -0.5 * matmul(mean, matmul(invCov, mean))
            let term3 = log(MLXArray(priors[i]))

            scores.append(term1 + term2 + term3)
        }

        let allScores = stacked(scores, axis: 1)
        let idx = argMax(allScores, axis: 1)

        let preds = idx.asArray(Int32.self).map { classes[Int($0)] }
        return MLXArray(preds).reshaped([X.shape[0], 1])
    }
}

extension LDA: FittedStatePersistable {
    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let classes: [Float]
        public let means: [SwiftMachinaArray]
        public let priors: [Float]
        public let invCov: SwiftMachinaArray
    }

    public func fittedState() throws -> FittedState {
        guard let invCov else {
            throw SwiftMachinaError.notFitted("LDA must be fitted before saving")
        }

        return FittedState(
            schemaVersion: fittedStateSchemaVersion,
            modelType: "LDA",
            classes: classes,
            means: means.map(SwiftMachinaArray.init),
            priors: priors,
            invCov: SwiftMachinaArray(invCov)
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "LDA"
        )
        try require(!fittedState.classes.isEmpty, .notFitted("LDA fitted state must contain classes"))
        try require(
            fittedState.means.count == fittedState.classes.count &&
            fittedState.priors.count == fittedState.classes.count,
            .invalidShape("LDA fitted state arrays must match class count")
        )

        let invCov = try fittedState.invCov.mlxArray()
        try require(
            invCov.shape.count == 2 && invCov.shape[0] == invCov.shape[1] && invCov.shape[0] > 0,
            .invalidShape("LDA invCov must be a square 2D matrix")
        )
        let means = try fittedState.means.map { try $0.mlxArray() }
        for mean in means {
            try require(
                mean.shape == [invCov.shape[0]],
                .invalidShape("LDA means must match the invCov dimension")
            )
        }

        self.classes = fittedState.classes
        self.means = means
        self.priors = fittedState.priors
        self.invCov = invCov
    }
}
