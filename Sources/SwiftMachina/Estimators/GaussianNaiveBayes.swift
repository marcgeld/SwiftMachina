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
    private var nFeatures: Int = 0

    private let epsilon: Float = 1e-8

    public init() {}

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))

        let yFlat = y.flattened()
        classes = Array(Set(yFlat.asArray(Float.self))).sorted()
        means = []
        variances = []
        priors = []
        nFeatures = X.shape[1]

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
    public func predict(X: MLXArray) throws -> MLXArray {
        try require(!classes.isEmpty, .notFitted("GaussianNaiveBayes must be fitted before prediction"))
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == nFeatures, .invalidShape("X must have the same number of features as training data"))

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

extension GaussianNaiveBayes: FittedStatePersistable {
    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let classes: [Float]
        public let means: [SwiftMachinaArray]
        public let variances: [SwiftMachinaArray]
        public let priors: [Float]
        public let nFeatures: Int
        public let epsilon: Float
    }

    public func fittedState() throws -> FittedState {
        try require(!classes.isEmpty, .notFitted("GaussianNaiveBayes must be fitted before saving"))
        return FittedState(
            schemaVersion: fittedStateSchemaVersion,
            modelType: "GaussianNaiveBayes",
            classes: classes,
            means: means.map(SwiftMachinaArray.init),
            variances: variances.map(SwiftMachinaArray.init),
            priors: priors,
            nFeatures: nFeatures,
            epsilon: epsilon
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "GaussianNaiveBayes"
        )
        try require(!fittedState.classes.isEmpty, .notFitted("GaussianNaiveBayes fitted state must contain classes"))
        try require(fittedState.nFeatures > 0, .invalidShape("nFeatures must be greater than zero"))
        try require(
            fittedState.means.count == fittedState.classes.count &&
            fittedState.variances.count == fittedState.classes.count &&
            fittedState.priors.count == fittedState.classes.count,
            .invalidShape("GaussianNaiveBayes fitted state arrays must match class count")
        )

        let means = try fittedState.means.map { try $0.mlxArray() }
        let variances = try fittedState.variances.map { try $0.mlxArray() }
        for array in means + variances {
            try require(
                array.shape == [fittedState.nFeatures],
                .invalidShape("GaussianNaiveBayes means and variances must have shape [nFeatures]")
            )
        }

        self.classes = fittedState.classes
        self.means = means
        self.variances = variances
        self.priors = fittedState.priors
        self.nFeatures = fittedState.nFeatures
    }
}
