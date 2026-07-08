//
//  DecisionTree.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import Foundation
import MLX

// MARK: - Decision Tree CART Classifier

public struct DecisionTree: Classifier {

    public final class Node {
        let prediction: Float
        let feature: Int?
        let threshold: Float?
        let left: Node?
        let right: Node?

        var isLeaf: Bool { feature == nil }

        init(prediction: Float) {
            self.prediction = prediction
            self.feature = nil
            self.threshold = nil
            self.left = nil
            self.right = nil
        }

        init(
            prediction: Float,
            feature: Int,
            threshold: Float,
            left: Node,
            right: Node
        ) {
            self.prediction = prediction
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    private var root: Node?

    public let maxDepth: Int
    public let minSamplesSplit: Int
    public let minSamplesLeaf: Int
    public let minImpurityDecrease: Float
    public let maxFeatures: Int?
    public let randomThresholds: Bool
    public let randomState: UInt64?

    private var nFeatures: Int = 0
    private var classValues: [Float] = []
    private var classIndex: [Float: Int] = [:]

    public init(
        maxDepth: Int = 5,
        minSamplesSplit: Int = 2,
        minSamplesLeaf: Int = 1,
        minImpurityDecrease: Float = 0.0,
        maxFeatures: Int? = nil,
        randomThresholds: Bool = false,
        randomState: UInt64? = nil
    ) throws {
        try require(maxDepth >= 0, .invalidParameter("maxDepth must be non-negative"))
        try require(minSamplesSplit >= 2, .invalidParameter("minSamplesSplit must be at least 2"))
        try require(minSamplesLeaf >= 1, .invalidParameter("minSamplesLeaf must be at least 1"))
        if let maxFeatures {
            try require(maxFeatures > 0, .invalidParameter("maxFeatures must be greater than zero"))
        }
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minImpurityDecrease = minImpurityDecrease
        self.maxFeatures = maxFeatures
        self.randomThresholds = randomThresholds
        self.randomState = randomState
    }

    // MARK: - Fit

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])

        let nSamples = X.shape[0]
        try require(nSamples > 0, .invalidShape("X must contain at least one sample"))
        nFeatures = X.shape[1]

        let xData = X.asArray(Float.self)
        let yData = y.flattened().asArray(Float.self)

        classValues = Array(Set(yData)).sorted()
        classIndex = Dictionary(uniqueKeysWithValues: classValues.enumerated().map { ($1, $0) })

        let indices = Array(0..<nSamples)
        var rng = randomState.map { SeededRandomNumberGenerator(seed: $0) }
        root = buildTree(
            xData: xData,
            yData: yData,
            indices: indices,
            depth: 0,
            rng: &rng
        )
    }

    // MARK: - Predict

    public func predict(X: MLXArray) throws -> MLXArray {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        guard let root else {
            throw SwiftMachinaError.notFitted("DecisionTree must be fitted before prediction")
        }
        try require(X.shape[1] == nFeatures, .invalidShape("X must have the same number of features as training data"))

        let rows = X.shape[0]
        let cols = X.shape[1]
        let xData = X.asArray(Float.self)

        let predictions = (0..<rows).map { row in
            predictRow(
                xData: xData,
                row: row,
                cols: cols,
                node: root
            )
        }

        return MLXArray(predictions).reshaped([rows, 1])
    }

    // MARK: - Tree Building

    private func buildTree(
        xData: [Float],
        yData: [Float],
        indices: [Int],
        depth: Int,
        rng: inout SeededRandomNumberGenerator?
    ) -> Node {

        let counts = classCounts(yData: yData, indices: indices)
        let prediction = majorityClass(from: counts)

        if shouldStop(indices: indices, counts: counts, depth: depth) {
            return Node(prediction: prediction)
        }

        guard let split = bestSplit(
            xData: xData,
            yData: yData,
            indices: indices,
            parentCounts: counts,
            rng: &rng
        ) else {
            return Node(prediction: prediction)
        }

        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(indices.count)
        rightIndices.reserveCapacity(indices.count)

        for row in indices {
            let value = xData[row * nFeatures + split.feature]
            if value <= split.threshold {
                leftIndices.append(row)
            } else {
                rightIndices.append(row)
            }
        }

        if leftIndices.count < minSamplesLeaf || rightIndices.count < minSamplesLeaf {
            return Node(prediction: prediction)
        }

        let left = buildTree(
            xData: xData,
            yData: yData,
            indices: leftIndices,
            depth: depth + 1,
            rng: &rng
        )

        let right = buildTree(
            xData: xData,
            yData: yData,
            indices: rightIndices,
            depth: depth + 1,
            rng: &rng
        )

        return Node(
            prediction: prediction,
            feature: split.feature,
            threshold: split.threshold,
            left: left,
            right: right
        )
    }

    private func shouldStop(
        indices: [Int],
        counts: [Int],
        depth: Int
    ) -> Bool {
        if depth >= maxDepth { return true }
        if indices.count < minSamplesSplit { return true }
        if counts.filter({ $0 > 0 }).count <= 1 { return true }
        return false
    }

    // MARK: - Best Split

    private struct Split {
        let feature: Int
        let threshold: Float
        let impurityDecrease: Float
    }

    private func bestSplit(
        xData: [Float],
        yData: [Float],
        indices: [Int],
        parentCounts: [Int],
        rng: inout SeededRandomNumberGenerator?
    ) -> Split? {

        let parentImpurity = gini(counts: parentCounts)
        let parentN = Float(indices.count)

        var bestFeature: Int?
        var bestThreshold: Float = 0
        var bestDecrease: Float = 0

        for feature in featureCandidates(rng: &rng) {

            var sortedRows = indices.map { row -> (row: Int, value: Float, labelIndex: Int) in
                let label = yData[row]
                return (
                    row: row,
                    value: xData[row * nFeatures + feature],
                    labelIndex: classIndex[label]!
                )
            }

            sortedRows.sort { $0.value < $1.value }

            if randomThresholds {
                let lower = sortedRows.first!.value
                let upper = sortedRows.last!.value
                if lower == upper {
                    continue
                }

                let threshold = randomThreshold(lower: lower, upper: upper, rng: &rng)
                let candidate = evaluateSplit(
                    sortedRows: sortedRows,
                    feature: feature,
                    threshold: threshold,
                    parentImpurity: parentImpurity,
                    parentN: parentN
                )

                if let candidate, candidate.impurityDecrease > bestDecrease {
                    bestDecrease = candidate.impurityDecrease
                    bestFeature = candidate.feature
                    bestThreshold = candidate.threshold
                }
                continue
            }

            var leftCounts = Array(repeating: 0, count: classValues.count)
            var rightCounts = parentCounts

            for i in 0..<(sortedRows.count - 1) {
                let labelIdx = sortedRows[i].labelIndex
                leftCounts[labelIdx] += 1
                rightCounts[labelIdx] -= 1

                let leftN = i + 1
                let rightN = sortedRows.count - leftN

                if leftN < minSamplesLeaf || rightN < minSamplesLeaf {
                    continue
                }

                let currentValue = sortedRows[i].value
                let nextValue = sortedRows[i + 1].value

                // No useful threshold between equal feature values.
                if currentValue == nextValue {
                    continue
                }

                let leftImpurity = gini(counts: leftCounts)
                let rightImpurity = gini(counts: rightCounts)

                let weightedImpurity =
                    (Float(leftN) / parentN) * leftImpurity +
                    (Float(rightN) / parentN) * rightImpurity

                let decrease = parentImpurity - weightedImpurity

                if decrease > bestDecrease {
                    bestDecrease = decrease
                    bestFeature = feature
                    bestThreshold = (currentValue + nextValue) / 2.0
                }
            }
        }

        guard let feature = bestFeature else {
            return nil
        }

        guard bestDecrease >= minImpurityDecrease else {
            return nil
        }

        return Split(
            feature: feature,
            threshold: bestThreshold,
            impurityDecrease: bestDecrease
        )
    }

    private func featureCandidates(rng: inout SeededRandomNumberGenerator?) -> [Int] {
        var features = Array(0..<nFeatures)

        guard let maxFeatures else {
            return features
        }

        let cappedMaxFeatures = min(max(maxFeatures, 1), nFeatures)
        guard cappedMaxFeatures < nFeatures else {
            return features
        }

        if var seeded = rng {
            features.shuffle(using: &seeded)
            rng = seeded
        } else {
            features.shuffle()
        }

        return Array(features.prefix(cappedMaxFeatures))
    }

    private func randomThreshold(
        lower: Float,
        upper: Float,
        rng: inout SeededRandomNumberGenerator?
    ) -> Float {
        if var seeded = rng {
            let value = Float.random(in: lower..<upper, using: &seeded)
            rng = seeded
            return value
        }

        return Float.random(in: lower..<upper)
    }

    private func evaluateSplit(
        sortedRows: [(row: Int, value: Float, labelIndex: Int)],
        feature: Int,
        threshold: Float,
        parentImpurity: Float,
        parentN: Float
    ) -> Split? {
        var leftCounts = Array(repeating: 0, count: classValues.count)
        var rightCounts = Array(repeating: 0, count: classValues.count)
        var leftN = 0
        var rightN = 0

        for row in sortedRows {
            if row.value <= threshold {
                leftCounts[row.labelIndex] += 1
                leftN += 1
            } else {
                rightCounts[row.labelIndex] += 1
                rightN += 1
            }
        }

        guard leftN >= minSamplesLeaf, rightN >= minSamplesLeaf else {
            return nil
        }

        let leftImpurity = gini(counts: leftCounts)
        let rightImpurity = gini(counts: rightCounts)
        let weightedImpurity =
            (Float(leftN) / parentN) * leftImpurity +
            (Float(rightN) / parentN) * rightImpurity

        return Split(
            feature: feature,
            threshold: threshold,
            impurityDecrease: parentImpurity - weightedImpurity
        )
    }

    // MARK: - Impurity / Counts

    private func classCounts(
        yData: [Float],
        indices: [Int]
    ) -> [Int] {
        var counts = Array(repeating: 0, count: classValues.count)

        for row in indices {
            let label = yData[row]
            counts[classIndex[label]!] += 1
        }

        return counts
    }

    private func gini(counts: [Int]) -> Float {
        let total = counts.reduce(0, +)
        guard total > 0 else { return 0 }

        let totalF = Float(total)
        var sum: Float = 0

        for count in counts {
            let p = Float(count) / totalF
            sum += p * p
        }

        return 1.0 - sum
    }

    private func majorityClass(from counts: [Int]) -> Float {
        let bestIndex = counts.indices.max { counts[$0] < counts[$1] }!
        return classValues[bestIndex]
    }

    // MARK: - Row Prediction

    private func predictRow(
        xData: [Float],
        row: Int,
        cols: Int,
        node: Node
    ) -> Float {
        var current = node

        while let feature = current.feature,
              let threshold = current.threshold {

            let value = xData[row * cols + feature]

            if value <= threshold {
                current = current.left!
            } else {
                current = current.right!
            }
        }

        return current.prediction
    }
}

extension DecisionTree: FittedStatePersistable {
    public final class NodeState: Codable {
        public let prediction: Float
        public let feature: Int?
        public let threshold: Float?
        public let left: NodeState?
        public let right: NodeState?

        public init(
            prediction: Float,
            feature: Int? = nil,
            threshold: Float? = nil,
            left: NodeState? = nil,
            right: NodeState? = nil
        ) {
            self.prediction = prediction
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let maxDepth: Int
        public let minSamplesSplit: Int
        public let minSamplesLeaf: Int
        public let minImpurityDecrease: Float
        public let maxFeatures: Int?
        public let randomThresholds: Bool
        public let randomState: UInt64?
        public let nFeatures: Int
        public let classValues: [Float]
        public let root: NodeState
    }

    public func fittedState() throws -> FittedState {
        guard let root else {
            throw SwiftMachinaError.notFitted("DecisionTree must be fitted before saving")
        }

        return FittedState(
            schemaVersion: 1,
            modelType: "DecisionTree",
            maxDepth: maxDepth,
            minSamplesSplit: minSamplesSplit,
            minSamplesLeaf: minSamplesLeaf,
            minImpurityDecrease: minImpurityDecrease,
            maxFeatures: maxFeatures,
            randomThresholds: randomThresholds,
            randomState: randomState,
            nFeatures: nFeatures,
            classValues: classValues,
            root: Self.encode(root)
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "DecisionTree"
        )
        try self.init(
            maxDepth: fittedState.maxDepth,
            minSamplesSplit: fittedState.minSamplesSplit,
            minSamplesLeaf: fittedState.minSamplesLeaf,
            minImpurityDecrease: fittedState.minImpurityDecrease,
            maxFeatures: fittedState.maxFeatures,
            randomThresholds: fittedState.randomThresholds,
            randomState: fittedState.randomState
        )
        try require(fittedState.nFeatures > 0, .invalidShape("nFeatures must be greater than zero"))
        try require(!fittedState.classValues.isEmpty, .notFitted("DecisionTree fitted state must contain class values"))

        self.nFeatures = fittedState.nFeatures
        self.classValues = fittedState.classValues
        self.classIndex = Dictionary(uniqueKeysWithValues: fittedState.classValues.enumerated().map { ($1, $0) })
        self.root = try Self.decode(fittedState.root)
    }

    private static func encode(_ node: Node) -> NodeState {
        NodeState(
            prediction: node.prediction,
            feature: node.feature,
            threshold: node.threshold,
            left: node.left.map(encode),
            right: node.right.map(encode)
        )
    }

    private static func decode(_ state: NodeState) throws -> Node {
        if let feature = state.feature,
           let threshold = state.threshold,
           let left = state.left,
           let right = state.right {
            return Node(
                prediction: state.prediction,
                feature: feature,
                threshold: threshold,
                left: try decode(left),
                right: try decode(right)
            )
        }

        try require(
            state.feature == nil && state.threshold == nil && state.left == nil && state.right == nil,
            .invalidShape("DecisionTree node must be either a leaf or a complete split")
        )
        return Node(prediction: state.prediction)
    }
}
