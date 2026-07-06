//
//  RandomForest.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

public struct RandomForest: Classifier {

    private var trees: [DecisionTree] = []
    public let nTrees: Int
    public let maxDepth: Int
    public let randomState: UInt64?

    public init(nTrees: Int = 10, maxDepth: Int = 5, randomState: UInt64? = nil) {
        precondition(nTrees > 0, "nTrees must be greater than zero")
        precondition(maxDepth >= 0, "maxDepth must be non-negative")
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.randomState = randomState
    }

    public mutating func fit(X: MLXArray, y: MLXArray) {
        precondition(X.shape.count == 2, "X must be a 2D array")
        precondition(y.shape[0] == X.shape[0], "X and y must have same number of rows")

        trees = []
        var rng = randomState.map { SeededRandomNumberGenerator(seed: $0) }

        for _ in 0..<nTrees {

            let bootIdx = MLXArray((0..<X.shape[0]).map { _ in
                Int32(Self.randomInt(in: 0..<X.shape[0], rng: &rng))
            })
            let Xb = X[bootIdx]
            let yb = y[bootIdx]

            var tree = DecisionTree(maxDepth: maxDepth)
            tree.fit(X: Xb, y: yb)

            trees.append(tree)
        }
    }

    public func predict(X: MLXArray) -> MLXArray {
        precondition(!trees.isEmpty, "RandomForest must be fitted before prediction")

        let preds = trees.map { $0.predict(X: X) }
        let stacked = MLX.stacked(preds, axis: 1)

        let mean = stacked.mean(axis: 1)

        return `where`(mean .> 0.5, MLXArray(1), MLXArray(0))
            .reshaped([X.shape[0], 1])
    }

    private static func randomInt(
        in range: Range<Int>,
        rng: inout SeededRandomNumberGenerator?
    ) -> Int {
        if var seeded = rng {
            let value = Int.random(in: range, using: &seeded)
            rng = seeded
            return value
        }
        return Int.random(in: range)
    }
}
