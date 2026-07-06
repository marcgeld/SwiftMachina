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

    public init(regParam: Float = 0.0) {
        self.regParam = regParam
    }

    // MARK: - Fit

    public mutating func fit(X: MLXArray, y: MLXArray) {

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()

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

    public func predict(X: MLXArray) -> MLXArray {

        var scores: [MLXArray] = []

        for i in 0..<classes.count {

            let mean = means[i]
            let cov = covs[i]

            let invCov = inv(cov, stream: .cpu)

            // log(det(cov)) via Cholesky: 2 * sum(log(diag(L)))
            let L = cholesky(cov, stream: .cpu)
            let logDet = 2 * log(L.diagonal()).sum()

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
