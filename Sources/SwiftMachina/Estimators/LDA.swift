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

    public mutating func fit(X: MLXArray, y: MLXArray) {

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()

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

    public func predict(X: MLXArray) -> MLXArray {

        guard let invCov else {
            fatalError("LDA not fitted")
        }

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
