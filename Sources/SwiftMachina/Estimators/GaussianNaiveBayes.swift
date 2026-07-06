//
//  GaussianNaiveBayes.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//


import MLX

// MARK: - Gaussian Naive Bayes
//
// Assumes features are independent and normally distributed.
public struct GaussianNaiveBayes: Classifier {

    private var classes: [Float] = []
    private var means: [MLXArray] = []
    private var variances: [MLXArray] = []
    private var priors: [Float] = []

    private let epsilon: Float = 1e-8

    public init() {}

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) {

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()

        for c in classes {

            let idx = maskToIndices(yFlat .== c)
            let Xc = X[idx]

            let mean = Xc.mean(axis: 0)
            let variance = ((Xc - mean) * (Xc - mean)).mean(axis: 0) + epsilon

            let prior = Float(Xc.shape[0]) / Float(X.shape[0])

            means.append(mean)
            variances.append(variance)
            priors.append(prior)
        }
    }

    // MARK: - Predict
    public func predict(X: MLXArray) -> MLXArray {

        var scores: [MLXArray] = []

        for i in 0..<classes.count {

            let mean = means[i]
            let varc = variances[i]

            let logLikelihood =
                -0.5 * log(2 * Float.pi * varc)
                - ((X - mean) * (X - mean)) / (2 * varc)

            let total = logLikelihood.sum(axis: 1)
            let logPrior = log(MLXArray(priors[i]))

            scores.append(total + logPrior)
        }

        let allScores = stacked(scores, axis: 1)
        let idx = argMax(allScores, axis: 1)

        let preds = idx.asArray(Int32.self).map { classes[Int($0)] }
        return MLXArray(preds).reshaped([X.shape[0], 1])
    }
}
