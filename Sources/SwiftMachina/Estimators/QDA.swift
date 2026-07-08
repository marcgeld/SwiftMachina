//
//  QDA.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Quadratic Discriminant Analysis
//
// Each class has its own covariance → quadratic decision boundary.
public struct QDA: Classifier {

    private var classes: [Float] = []
    private var means: [MLXArray] = []
    private var covs: [MLXArray] = []
    private var priors: [Float] = []
    private let regParam: Float

    public init(regParam: Float = 0.0) throws {
        try require(regParam >= 0 && regParam <= 1, .invalidParameter("regParam must be in [0, 1]"))
        self.regParam = regParam
    }

    // MARK: - Fit

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()
        means = []
        covs = []
        priors = []

        for c in classes {

            let idx = maskToIndices(yFlat .== c)
            let Xc = X[idx]

            let mean = Xc.mean(axis: 0)
            means.append(mean)

            let centered = Xc - mean
            let cov = matmul(centered.T, centered) / Float(Xc.shape[0])

            let eye = MLXArray.eye(Xc.shape[1])
            let regCov = (1 - regParam) * cov + regParam * eye + Float(1e-6) * eye
            covs.append(regCov)

            let prior = Float(Xc.shape[0]) / Float(X.shape[0])
            priors.append(prior)
        }
    }

    // MARK: - Predict

    public func predict(X: MLXArray) throws -> MLXArray {
        try require(!classes.isEmpty, .notFitted("QDA must be fitted before prediction"))
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == covs[0].shape[0], .invalidShape("X must have the same number of features as training data"))

        var scores: [MLXArray] = []

        for i in 0..<classes.count {

            let mean = means[i]
            let cov = covs[i]

            // SwiftNumerica works in Double precision; MLX (Float32, also
            // on the CPU stream) is the fallback for degenerate input.
            let invCov = numericaInverse(cov) ?? inv(cov, stream: .cpu)

            // log(det(cov)) via sum of log-eigenvalues, falling back to
            // Cholesky: 2 * sum(log(diag(L)))
            let logDet: MLXArray
            if let value = numericaLogDeterminant(symmetric: cov) {
                logDet = MLXArray(value)
            } else {
                let L = cholesky(cov, stream: .cpu)
                logDet = 2 * log(L.diagonal()).sum()
            }

            let diff = X - mean

            let quad =
                (matmul(diff, invCov) * diff).sum(axis: 1)

            let term =
                -0.5 * quad
                - 0.5 * logDet
                + log(MLXArray(priors[i]))

            scores.append(term)
        }

        let allScores = stacked(scores, axis: 1)
        let idx = argMax(allScores, axis: 1)

        let preds = idx.asArray(Int32.self).map { classes[Int($0)] }
        return MLXArray(preds).reshaped([X.shape[0], 1])
    }
}
