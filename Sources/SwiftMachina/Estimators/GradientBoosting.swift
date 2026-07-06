//
//  GradientBoosting.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

public struct GradientBoosting: Classifier {

    private var trees: [DecisionTree] = []
    public let nEstimators: Int
    public let learningRate: Float

    public init(nEstimators: Int = 10, learningRate: Float = 0.1) {
        self.nEstimators = nEstimators
        self.learningRate = learningRate
    }

    public mutating func fit(X: MLXArray, y: MLXArray) {

        var residual = y
        trees = []

        for _ in 0..<nEstimators {

            var tree = DecisionTree(maxDepth: 3)
            tree.fit(X: X, y: residual)

            let pred = tree.predict(X: X)
            residual = residual - learningRate * pred

            trees.append(tree)
        }
    }

    public func predict(X: MLXArray) -> MLXArray {

        var sum = MLXArray.zeros([X.shape[0], 1])

        for tree in trees {
            sum += learningRate * tree.predict(X: X)
        }

        return `where`(sum .> 0.5, MLXArray(1), MLXArray(0))
    }
}
