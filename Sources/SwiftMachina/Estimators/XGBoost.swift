//
//  XGBoost.swift
//  SwiftMachina
//

import Foundation
import MLX

// MARK: - XGBoost-style Gradient Boosting (Binary)
//
// Regularized second-order gradient boosting in the style of XGBoost:
//
//  - Logistic loss with gradient g = p - y and hessian h = p(1 - p)
//  - Leaf weights w = -softThreshold(G, alpha) / (H + lambda)
//  - Split gain = 0.5 * (score(L) + score(R) - score(parent)) - gamma
//  - Histogram-based split finding over quantile bins
//  - Shrinkage (learningRate), row subsampling, and per-tree column
//    subsampling, seeded via randomState
//  - Optional early stopping against an evaluation set (log loss)
//
// Reference: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System"
// https://arxiv.org/abs/1603.02754

public struct XGBoostClassifier: Classifier {

    // MARK: - Hyperparameters

    public let nEstimators: Int
    public let learningRate: Float
    public let maxDepth: Int
    /// Minimum hessian sum required in each child (min_child_weight).
    public let minChildWeight: Float
    /// L2 regularization on leaf weights (lambda).
    public let lambda: Float
    /// L1 regularization on leaf weights (alpha).
    public let alpha: Float
    /// Minimum loss reduction required to make a split (gamma).
    public let gamma: Float
    /// Fraction of rows sampled per tree.
    public let subsample: Float
    /// Fraction of features sampled per tree.
    public let colsampleByTree: Float
    /// Maximum number of histogram bins per feature.
    public let maxBins: Int
    public let randomState: UInt64?

    // MARK: - Fitted state

    private var trees: [BoostedTree] = []
    private var classValues: [Float] = []
    private var baseScore: Float = 0
    private var nFeatures: Int = 0

    /// The best boosting round found by early stopping (0-based),
    /// or nil when early stopping was not used.
    public private(set) var bestIteration: Int?

    /// Number of boosting rounds actually kept after fitting.
    public var boostedRounds: Int { trees.count }

    // MARK: - Init

    public init(
        nEstimators: Int = 100,
        learningRate: Float = 0.3,
        maxDepth: Int = 6,
        minChildWeight: Float = 1.0,
        lambda: Float = 1.0,
        alpha: Float = 0.0,
        gamma: Float = 0.0,
        subsample: Float = 1.0,
        colsampleByTree: Float = 1.0,
        maxBins: Int = 256,
        randomState: UInt64? = nil
    ) throws {
        try require(nEstimators > 0, .invalidParameter("nEstimators must be greater than zero"))
        try require(learningRate > 0, .invalidParameter("learningRate must be greater than zero"))
        try require(maxDepth >= 0, .invalidParameter("maxDepth must be non-negative"))
        try require(minChildWeight >= 0, .invalidParameter("minChildWeight must be non-negative"))
        try require(lambda >= 0, .invalidParameter("lambda must be non-negative"))
        try require(alpha >= 0, .invalidParameter("alpha must be non-negative"))
        try require(gamma >= 0, .invalidParameter("gamma must be non-negative"))
        try require(subsample > 0 && subsample <= 1, .invalidParameter("subsample must be in (0, 1]"))
        try require(colsampleByTree > 0 && colsampleByTree <= 1, .invalidParameter("colsampleByTree must be in (0, 1]"))
        try require(maxBins >= 2 && maxBins <= 65_535, .invalidParameter("maxBins must be in [2, 65535]"))
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.lambda = lambda
        self.alpha = alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsampleByTree = colsampleByTree
        self.maxBins = maxBins
        self.randomState = randomState
    }

    // MARK: - Fit (protocol conformance)

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try fit(X: X, y: y, evalX: nil, evalY: nil, earlyStoppingRounds: nil)
    }

    // MARK: - Fit

    public mutating func fit(
        X: MLXArray,
        y: MLXArray,
        evalX: MLXArray?,
        evalY: MLXArray?,
        earlyStoppingRounds: Int?
    ) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(y.shape[0] == X.shape[0], .invalidShape("X and y must have same number of rows"))
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))
        if earlyStoppingRounds != nil {
            try require(earlyStoppingRounds! > 0, .invalidParameter("earlyStoppingRounds must be greater than zero"))
            try require(evalX != nil && evalY != nil, .invalidParameter("earlyStoppingRounds requires evalX and evalY"))
        }

        let rows = X.shape[0]
        nFeatures = X.shape[1]
        let xData = X.asArray(Float.self)
        let yLabels = y.flattened().asArray(Float.self)

        classValues = Array(Set(yLabels)).sorted()
        try require(classValues.count == 2, .unsupported("XGBoostClassifier supports binary classification"))
        let encodedY = yLabels.map { $0 == classValues[1] ? Float(1) : Float(0) }

        if let evalX {
            try require(evalX.shape.count == 2 && evalX.shape[1] == nFeatures,
                        .invalidShape("evalX must have the same number of features as X"))
            try require(evalY!.shape[0] == evalX.shape[0],
                        .invalidShape("evalX and evalY must have same number of rows"))
        }

        let positiveRate = Self.clamp(encodedY.reduce(0, +) / Float(rows), min: 1e-6, max: 1 - 1e-6)
        baseScore = log(positiveRate / (1 - positiveRate))
        trees = []
        bestIteration = nil

        // Bin features once: quantile cut points, then row-major bin indices.
        let cuts = Self.quantileCuts(xData: xData, rows: rows, cols: nFeatures, maxBins: maxBins)
        let bins = Self.binRows(xData: xData, rows: rows, cols: nFeatures, cuts: cuts)

        var rng = randomState.map { SeededRandomNumberGenerator(seed: $0) }
        var margins = Array(repeating: baseScore, count: rows)

        // Early stopping state
        let evalData = evalX.map { ($0.asArray(Float.self), $0.shape[0]) }
        let evalTargets = evalY.map { array in
            array.flattened().asArray(Float.self).map { $0 == classValues[1] ? Float(1) : Float(0) }
        }
        var evalMargins = evalData.map { Array(repeating: baseScore, count: $0.1) }
        var bestLoss = Float.infinity
        var bestRound = -1

        let builder = TreeBuilder(
            maxDepth: maxDepth,
            minChildWeight: minChildWeight,
            lambda: lambda,
            alpha: alpha,
            gamma: gamma
        )

        for round in 0..<nEstimators {

            // Second-order statistics of the logistic loss
            var g = [Double](repeating: 0, count: rows)
            var h = [Double](repeating: 0, count: rows)
            for i in 0..<rows {
                let p = Double(Self.sigmoid(margins[i]))
                g[i] = p - Double(encodedY[i])
                h[i] = max(p * (1 - p), 1e-16)
            }

            let sampledRows = sampleIndices(count: rows, fraction: subsample, rng: &rng)
            let sampledFeatures = sampleIndices(count: nFeatures, fraction: colsampleByTree, rng: &rng)

            let tree = builder.build(
                bins: bins,
                cuts: cuts,
                g: g,
                h: h,
                indices: sampledRows,
                features: sampledFeatures,
                cols: nFeatures
            )
            trees.append(tree)

            for i in 0..<rows {
                margins[i] += learningRate * tree.predictRow(xData: xData, row: i, cols: nFeatures)
            }

            // Early stopping on eval log loss
            if let earlyStoppingRounds, let (evalX, evalRows) = evalData, let targets = evalTargets {
                for i in 0..<evalRows {
                    evalMargins![i] += learningRate * tree.predictRow(xData: evalX, row: i, cols: nFeatures)
                }
                let loss = Self.logLoss(margins: evalMargins!, targets: targets)
                if loss < bestLoss {
                    bestLoss = loss
                    bestRound = round
                }
                if round - bestRound >= earlyStoppingRounds {
                    trees.removeLast(trees.count - (bestRound + 1))
                    bestIteration = bestRound
                    return
                }
            }
        }

        if evalData != nil {
            bestIteration = bestRound
        }
    }

    // MARK: - Predict probabilities

    public func predictProba(X: MLXArray) throws -> MLXArray {
        let margins = try rawMargins(X: X)
        let probabilities = margins.map(Self.sigmoid)
        return MLXArray(probabilities).reshaped([X.shape[0], 1])
    }

    // MARK: - Predict classes

    public func predict(X: MLXArray) throws -> MLXArray {
        let margins = try rawMargins(X: X)
        let predictions = margins.map { $0 > 0 ? classValues[1] : classValues[0] }
        return MLXArray(predictions).reshaped([X.shape[0], 1])
    }

    private func rawMargins(X: MLXArray) throws -> [Float] {
        try require(!trees.isEmpty, .notFitted("XGBoostClassifier must be fitted before prediction"))
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == nFeatures, .invalidShape("X must have the same number of features as training data"))

        let rows = X.shape[0]
        let xData = X.asArray(Float.self)
        var margins = Array(repeating: baseScore, count: rows)

        for tree in trees {
            for i in 0..<rows {
                margins[i] += learningRate * tree.predictRow(xData: xData, row: i, cols: nFeatures)
            }
        }
        return margins
    }

    // MARK: - Sampling

    private func sampleIndices(
        count: Int,
        fraction: Float,
        rng: inout SeededRandomNumberGenerator?
    ) -> [Int] {
        guard fraction < 1 else { return Array(0..<count) }

        let sampleCount = max(1, Int((Float(count) * fraction).rounded()))
        var indices = Array(0..<count)
        if var seeded = rng {
            indices.shuffle(using: &seeded)
            rng = seeded
        } else {
            indices.shuffle()
        }
        return Array(indices.prefix(sampleCount)).sorted()
    }

    // MARK: - Binning

    /// Quantile cut points per feature. Going left at cut k means value <= cuts[k].
    private static func quantileCuts(
        xData: [Float],
        rows: Int,
        cols: Int,
        maxBins: Int
    ) -> [[Float]] {
        (0..<cols).map { feature in
            var sorted = (0..<rows).map { xData[$0 * cols + feature] }
            sorted.sort()

            var distinct: [Float] = []
            for value in sorted where value != distinct.last {
                distinct.append(value)
            }
            guard distinct.count > 1 else { return [] }

            if distinct.count <= maxBins {
                // Few distinct values: exact midpoints between neighbors.
                return (0..<(distinct.count - 1)).map { (distinct[$0] + distinct[$0 + 1]) / 2 }
            }

            var cuts: [Float] = []
            cuts.reserveCapacity(maxBins - 1)
            for i in 1..<maxBins {
                let idx = i * rows / maxBins
                let cut = (sorted[idx - 1] + sorted[idx]) / 2
                if cut != cuts.last {
                    cuts.append(cut)
                }
            }
            return cuts
        }
    }

    /// Row-major bin indices: bin = number of cuts strictly below the value.
    private static func binRows(
        xData: [Float],
        rows: Int,
        cols: Int,
        cuts: [[Float]]
    ) -> [UInt16] {
        var bins = [UInt16](repeating: 0, count: rows * cols)
        for feature in 0..<cols {
            let featureCuts = cuts[feature]
            guard !featureCuts.isEmpty else { continue }
            for row in 0..<rows {
                let value = xData[row * cols + feature]
                // Binary search: first cut >= value
                var low = 0
                var high = featureCuts.count
                while low < high {
                    let mid = (low + high) / 2
                    if featureCuts[mid] < value {
                        low = mid + 1
                    } else {
                        high = mid
                    }
                }
                bins[row * cols + feature] = UInt16(low)
            }
        }
        return bins
    }

    // MARK: - Scalar helpers

    private static func sigmoid(_ x: Float) -> Float {
        1 / (1 + exp(-x))
    }

    private static func clamp(_ value: Float, min lower: Float, max upper: Float) -> Float {
        Swift.max(lower, Swift.min(value, upper))
    }

    private static func logLoss(margins: [Float], targets: [Float]) -> Float {
        var total: Double = 0
        for (margin, target) in zip(margins, targets) {
            let p = Double(clamp(sigmoid(margin), min: 1e-7, max: 1 - 1e-7))
            total += Double(target) * Foundation.log(p) + (1 - Double(target)) * Foundation.log(1 - p)
        }
        return Float(-total / Double(margins.count))
    }
}

// MARK: - Tree Builder

private struct TreeBuilder {
    let maxDepth: Int
    let minChildWeight: Float
    let lambda: Float
    let alpha: Float
    let gamma: Float

    func build(
        bins: [UInt16],
        cuts: [[Float]],
        g: [Double],
        h: [Double],
        indices: [Int],
        features: [Int],
        cols: Int
    ) -> BoostedTree {
        let root = buildNode(
            bins: bins,
            cuts: cuts,
            g: g,
            h: h,
            indices: indices,
            features: features,
            cols: cols,
            depth: 0
        )
        return BoostedTree(root: root)
    }

    private func buildNode(
        bins: [UInt16],
        cuts: [[Float]],
        g: [Double],
        h: [Double],
        indices: [Int],
        features: [Int],
        cols: Int,
        depth: Int
    ) -> BoostedTree.Node {
        var totalG: Double = 0
        var totalH: Double = 0
        for row in indices {
            totalG += g[row]
            totalH += h[row]
        }

        let weight = leafWeight(g: totalG, h: totalH)

        if depth >= maxDepth || indices.count < 2 {
            return BoostedTree.Node(weight: weight)
        }

        guard let split = bestSplit(
            bins: bins,
            cuts: cuts,
            g: g,
            h: h,
            indices: indices,
            features: features,
            cols: cols,
            totalG: totalG,
            totalH: totalH
        ) else {
            return BoostedTree.Node(weight: weight)
        }

        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(indices.count)
        rightIndices.reserveCapacity(indices.count)
        for row in indices {
            if bins[row * cols + split.feature] <= split.bin {
                leftIndices.append(row)
            } else {
                rightIndices.append(row)
            }
        }

        let left = buildNode(
            bins: bins, cuts: cuts, g: g, h: h,
            indices: leftIndices, features: features, cols: cols, depth: depth + 1
        )
        let right = buildNode(
            bins: bins, cuts: cuts, g: g, h: h,
            indices: rightIndices, features: features, cols: cols, depth: depth + 1
        )

        return BoostedTree.Node(
            weight: weight,
            feature: split.feature,
            threshold: split.threshold,
            left: left,
            right: right
        )
    }

    private struct Split {
        let feature: Int
        let bin: UInt16
        let threshold: Float
        let gain: Double
    }

    private func bestSplit(
        bins: [UInt16],
        cuts: [[Float]],
        g: [Double],
        h: [Double],
        indices: [Int],
        features: [Int],
        cols: Int,
        totalG: Double,
        totalH: Double
    ) -> Split? {
        let parentScore = score(g: totalG, h: totalH)
        var best: Split?

        for feature in features {
            let featureCuts = cuts[feature]
            guard !featureCuts.isEmpty else { continue }

            let binCount = featureCuts.count + 1
            var histG = [Double](repeating: 0, count: binCount)
            var histH = [Double](repeating: 0, count: binCount)
            var histN = [Int](repeating: 0, count: binCount)

            for row in indices {
                let bin = Int(bins[row * cols + feature])
                histG[bin] += g[row]
                histH[bin] += h[row]
                histN[bin] += 1
            }

            var leftG: Double = 0
            var leftH: Double = 0
            var leftN = 0

            // Split at bin b sends bins 0...b left, so the last bin never splits.
            for bin in 0..<(binCount - 1) {
                leftG += histG[bin]
                leftH += histH[bin]
                leftN += histN[bin]

                let rightN = indices.count - leftN
                guard leftN > 0, rightN > 0 else { continue }

                let rightG = totalG - leftG
                let rightH = totalH - leftH
                guard leftH >= Double(minChildWeight), rightH >= Double(minChildWeight) else { continue }

                let gain = 0.5 * (score(g: leftG, h: leftH) + score(g: rightG, h: rightH) - parentScore) - Double(gamma)
                if gain > 0, gain > (best?.gain ?? 0) {
                    best = Split(
                        feature: feature,
                        bin: UInt16(bin),
                        threshold: featureCuts[bin],
                        gain: gain
                    )
                }
            }
        }

        return best
    }

    /// XGBoost structure score: softThreshold(G, alpha)^2 / (H + lambda)
    private func score(g: Double, h: Double) -> Double {
        let numerator = softThreshold(g)
        return numerator * numerator / (h + Double(lambda))
    }

    private func leafWeight(g: Double, h: Double) -> Float {
        Float(-softThreshold(g) / (h + Double(lambda)))
    }

    private func softThreshold(_ g: Double) -> Double {
        let a = Double(alpha)
        if g > a { return g - a }
        if g < -a { return g + a }
        return 0
    }
}

// MARK: - Boosted Tree

private struct BoostedTree {
    final class Node {
        let weight: Float
        let feature: Int?
        let threshold: Float?
        let left: Node?
        let right: Node?

        init(weight: Float) {
            self.weight = weight
            self.feature = nil
            self.threshold = nil
            self.left = nil
            self.right = nil
        }

        init(weight: Float, feature: Int, threshold: Float, left: Node, right: Node) {
            self.weight = weight
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    let root: Node

    func predictRow(xData: [Float], row: Int, cols: Int) -> Float {
        var current = root

        while let feature = current.feature,
              let threshold = current.threshold {
            let value = xData[row * cols + feature]
            current = value <= threshold ? current.left! : current.right!
        }

        return current.weight
    }
}
