//
//  GradientBoosting.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import Foundation
import MLX

public struct GradientBoosting: Classifier {

    private var trees: [RegressionTree] = []
    private var classValues: [Float] = []
    private var initialLogOdds: Float = 0
    private var nFeatures: Int = 0

    public let nEstimators: Int
    public let learningRate: Float
    public let maxDepth: Int

    public init(
        nEstimators: Int = 10,
        learningRate: Float = 0.1,
        maxDepth: Int = 3
    ) throws {
        try require(nEstimators > 0, .invalidParameter("nEstimators must be greater than zero"))
        try require(learningRate > 0, .invalidParameter("learningRate must be greater than zero"))
        try require(maxDepth >= 0, .invalidParameter("maxDepth must be non-negative"))
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
    }

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))

        let fittedFeatureCount = X.shape[1]
        let xData = X.asArray(Float.self)
        let yLabels = y.flattened().asArray(Float.self)
        let fittedClassValues = Array(Set(yLabels)).sorted()
        try require(fittedClassValues.count == 2, .unsupported("GradientBoosting supports binary classification"))

        let encodedY = yLabels.map { $0 == fittedClassValues[1] ? Float(1) : Float(0) }
        let positiveRate = Self.clamp(encodedY.reduce(0, +) / Float(encodedY.count), min: 1e-6, max: 1 - 1e-6)
        let fittedInitialLogOdds = log(positiveRate / (1 - positiveRate))

        nFeatures = fittedFeatureCount
        classValues = fittedClassValues
        initialLogOdds = fittedInitialLogOdds
        trees = []

        var rawScores = Array(repeating: fittedInitialLogOdds, count: X.shape[0])

        for _ in 0..<nEstimators {
            let probabilities = rawScores.map(Self.sigmoid)
            let residuals = zip(encodedY, probabilities).map { target, probability in
                target - probability
            }

            let tree = RegressionTree.fit(
                xData: xData,
                yData: residuals,
                rows: X.shape[0],
                cols: fittedFeatureCount,
                maxDepth: maxDepth
            )

            let updates = tree.predictRows(xData: xData, rows: X.shape[0], cols: fittedFeatureCount)
            for i in rawScores.indices {
                rawScores[i] += learningRate * updates[i]
            }

            trees.append(tree)
        }
    }

    public func predict(X: MLXArray) throws -> MLXArray {
        try require(!trees.isEmpty, .notFitted("GradientBoosting must be fitted before prediction"))
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try require(X.shape[1] == nFeatures, .invalidShape("X must have the same number of features as training data"))

        let xData = X.asArray(Float.self)
        var rawScores = Array(repeating: initialLogOdds, count: X.shape[0])

        for tree in trees {
            let updates = tree.predictRows(xData: xData, rows: X.shape[0], cols: nFeatures)
            for i in rawScores.indices {
                rawScores[i] += learningRate * updates[i]
            }
        }

        let predictions = rawScores.map { $0 > 0 ? classValues[1] : classValues[0] }
        return MLXArray(predictions).reshaped([X.shape[0], 1])
    }

    private static func sigmoid(_ x: Float) -> Float {
        1 / (1 + exp(-x))
    }

    private static func clamp(_ value: Float, min lower: Float, max upper: Float) -> Float {
        Swift.max(lower, Swift.min(value, upper))
    }
}

private struct RegressionTree {
    final class Node {
        let value: Float
        let feature: Int?
        let threshold: Float?
        let left: Node?
        let right: Node?

        init(value: Float) {
            self.value = value
            self.feature = nil
            self.threshold = nil
            self.left = nil
            self.right = nil
        }

        init(value: Float, feature: Int, threshold: Float, left: Node, right: Node) {
            self.value = value
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    let root: Node
    let maxDepth: Int
    let minSamplesLeaf: Int

    static func fit(
        xData: [Float],
        yData: [Float],
        rows: Int,
        cols: Int,
        maxDepth: Int
    ) -> RegressionTree {
        let tree = RegressionTree(
            root: buildTree(
                xData: xData,
                yData: yData,
                indices: Array(0..<rows),
                cols: cols,
                depth: 0,
                maxDepth: maxDepth,
                minSamplesLeaf: 1
            ),
            maxDepth: maxDepth,
            minSamplesLeaf: 1
        )
        return tree
    }

    func predictRows(xData: [Float], rows: Int, cols: Int) -> [Float] {
        (0..<rows).map { row in
            predictRow(xData: xData, row: row, cols: cols)
        }
    }

    private func predictRow(xData: [Float], row: Int, cols: Int) -> Float {
        var current = root

        while let feature = current.feature,
              let threshold = current.threshold {
            let value = xData[row * cols + feature]
            current = value <= threshold ? current.left! : current.right!
        }

        return current.value
    }

    private static func buildTree(
        xData: [Float],
        yData: [Float],
        indices: [Int],
        cols: Int,
        depth: Int,
        maxDepth: Int,
        minSamplesLeaf: Int
    ) -> Node {
        let value = mean(yData: yData, indices: indices)

        if depth >= maxDepth || indices.count <= minSamplesLeaf * 2 {
            return Node(value: value)
        }

        guard let split = bestSplit(
            xData: xData,
            yData: yData,
            indices: indices,
            cols: cols,
            minSamplesLeaf: minSamplesLeaf
        ) else {
            return Node(value: value)
        }

        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(indices.count)
        rightIndices.reserveCapacity(indices.count)

        for row in indices {
            if xData[row * cols + split.feature] <= split.threshold {
                leftIndices.append(row)
            } else {
                rightIndices.append(row)
            }
        }

        guard !leftIndices.isEmpty, !rightIndices.isEmpty else {
            return Node(value: value)
        }

        let left = buildTree(
            xData: xData,
            yData: yData,
            indices: leftIndices,
            cols: cols,
            depth: depth + 1,
            maxDepth: maxDepth,
            minSamplesLeaf: minSamplesLeaf
        )
        let right = buildTree(
            xData: xData,
            yData: yData,
            indices: rightIndices,
            cols: cols,
            depth: depth + 1,
            maxDepth: maxDepth,
            minSamplesLeaf: minSamplesLeaf
        )

        return Node(
            value: value,
            feature: split.feature,
            threshold: split.threshold,
            left: left,
            right: right
        )
    }

    private struct Split {
        let feature: Int
        let threshold: Float
        let loss: Float
    }

    private static func bestSplit(
        xData: [Float],
        yData: [Float],
        indices: [Int],
        cols: Int,
        minSamplesLeaf: Int
    ) -> Split? {
        var best: Split?

        for feature in 0..<cols {
            var rows = indices.map { row in
                (row: row, value: xData[row * cols + feature], target: yData[row])
            }
            rows.sort { $0.value < $1.value }

            let totalSum = rows.reduce(Float(0)) { $0 + $1.target }
            let totalSquaredSum = rows.reduce(Float(0)) { $0 + $1.target * $1.target }
            var leftSum: Float = 0
            var leftSquaredSum: Float = 0

            for i in 0..<(rows.count - 1) {
                leftSum += rows[i].target
                leftSquaredSum += rows[i].target * rows[i].target

                let leftN = i + 1
                let rightN = rows.count - leftN
                if leftN < minSamplesLeaf || rightN < minSamplesLeaf {
                    continue
                }

                let currentValue = rows[i].value
                let nextValue = rows[i + 1].value
                if currentValue == nextValue {
                    continue
                }

                let rightSum = totalSum - leftSum
                let rightSquaredSum = totalSquaredSum - leftSquaredSum
                let leftLoss = leftSquaredSum - (leftSum * leftSum / Float(leftN))
                let rightLoss = rightSquaredSum - (rightSum * rightSum / Float(rightN))
                let loss = leftLoss + rightLoss

                if best == nil || loss < best!.loss {
                    best = Split(
                        feature: feature,
                        threshold: (currentValue + nextValue) / 2,
                        loss: loss
                    )
                }
            }
        }

        return best
    }

    private static func mean(yData: [Float], indices: [Int]) -> Float {
        guard !indices.isEmpty else { return 0 }
        let sum = indices.reduce(Float(0)) { $0 + yData[$1] }
        return sum / Float(indices.count)
    }
}

extension GradientBoosting: FittedStatePersistable {
    public final class NodeState: Codable {
        public let value: Float
        public let feature: Int?
        public let threshold: Float?
        public let left: NodeState?
        public let right: NodeState?

        public init(
            value: Float,
            feature: Int? = nil,
            threshold: Float? = nil,
            left: NodeState? = nil,
            right: NodeState? = nil
        ) {
            self.value = value
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
        }
    }

    public struct RegressionTreeState: Codable {
        public let maxDepth: Int
        public let minSamplesLeaf: Int
        public let root: NodeState
    }

    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let nEstimators: Int
        public let learningRate: Float
        public let maxDepth: Int
        public let classValues: [Float]
        public let initialLogOdds: Float
        public let nFeatures: Int
        public let trees: [RegressionTreeState]
    }

    public func fittedState() throws -> FittedState {
        try require(!trees.isEmpty, .notFitted("GradientBoosting must be fitted before saving"))
        return FittedState(
            schemaVersion: 1,
            modelType: "GradientBoosting",
            nEstimators: nEstimators,
            learningRate: learningRate,
            maxDepth: maxDepth,
            classValues: classValues,
            initialLogOdds: initialLogOdds,
            nFeatures: nFeatures,
            trees: trees.map(Self.encode)
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "GradientBoosting"
        )
        try self.init(
            nEstimators: fittedState.nEstimators,
            learningRate: fittedState.learningRate,
            maxDepth: fittedState.maxDepth
        )
        try require(fittedState.classValues.count == 2, .unsupported("GradientBoosting supports binary classification"))
        try require(fittedState.nFeatures > 0, .invalidShape("nFeatures must be greater than zero"))
        try require(!fittedState.trees.isEmpty, .notFitted("GradientBoosting fitted state must contain trees"))

        self.classValues = fittedState.classValues
        self.initialLogOdds = fittedState.initialLogOdds
        self.nFeatures = fittedState.nFeatures
        self.trees = try fittedState.trees.map(Self.decode)
    }

    private static func encode(_ tree: RegressionTree) -> RegressionTreeState {
        RegressionTreeState(
            maxDepth: tree.maxDepth,
            minSamplesLeaf: tree.minSamplesLeaf,
            root: encode(tree.root)
        )
    }

    private static func encode(_ node: RegressionTree.Node) -> NodeState {
        NodeState(
            value: node.value,
            feature: node.feature,
            threshold: node.threshold,
            left: node.left.map(encode),
            right: node.right.map(encode)
        )
    }

    private static func decode(_ state: RegressionTreeState) throws -> RegressionTree {
        try require(state.maxDepth >= 0, .invalidParameter("maxDepth must be non-negative"))
        try require(state.minSamplesLeaf >= 1, .invalidParameter("minSamplesLeaf must be at least 1"))
        return RegressionTree(
            root: try decode(state.root),
            maxDepth: state.maxDepth,
            minSamplesLeaf: state.minSamplesLeaf
        )
    }

    private static func decode(_ state: NodeState) throws -> RegressionTree.Node {
        if let feature = state.feature,
           let threshold = state.threshold,
           let left = state.left,
           let right = state.right {
            return RegressionTree.Node(
                value: state.value,
                feature: feature,
                threshold: threshold,
                left: try decode(left),
                right: try decode(right)
            )
        }

        try require(
            state.feature == nil && state.threshold == nil && state.left == nil && state.right == nil,
            .invalidShape("GradientBoosting node must be either a leaf or a complete split")
        )
        return RegressionTree.Node(value: state.value)
    }
}
