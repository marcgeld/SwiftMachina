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

    public init(nTrees: Int = 10, maxDepth: Int = 5, randomState: UInt64? = nil) throws {
        try require(nTrees > 0, .invalidParameter("nTrees must be greater than zero"))
        try require(maxDepth >= 0, .invalidParameter("maxDepth must be non-negative"))
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.randomState = randomState
    }

    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
        try requireLabelVector(y, rows: X.shape[0])
        try require(X.shape[0] > 0, .invalidShape("X must contain at least one sample"))

        trees = []
        var rng = randomState.map { SeededRandomNumberGenerator(seed: $0) }

        for _ in 0..<nTrees {

            let bootIdx = MLXArray((0..<X.shape[0]).map { _ in
                Int32(Self.randomInt(in: 0..<X.shape[0], rng: &rng))
            })
            let Xb = X[bootIdx]
            let yb = y[bootIdx]

            var tree = try DecisionTree(maxDepth: maxDepth)
            try tree.fit(X: Xb, y: yb)

            trees.append(tree)
        }
    }

    public func predict(X: MLXArray) throws -> MLXArray {
        try require(!trees.isEmpty, .notFitted("RandomForest must be fitted before prediction"))

        let preds = try trees.map { try $0.predict(X: X) }
        return Self.majorityVote(preds, rows: X.shape[0])
    }

    private static func majorityVote(_ predictions: [MLXArray], rows: Int) -> MLXArray {
        let valuesByTree = predictions.map { $0.flattened().asArray(Float.self) }
        let voted = (0..<rows).map { row in
            var counts: [Float: Int] = [:]
            for values in valuesByTree {
                counts[values[row], default: 0] += 1
            }
            return counts.keys.sorted().max { lhs, rhs in
                let left = counts[lhs, default: 0]
                let right = counts[rhs, default: 0]
                return left == right ? lhs > rhs : left < right
            }!
        }
        return MLXArray(voted).reshaped([rows, 1])
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

extension RandomForest: FittedStatePersistable {
    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let nTrees: Int
        public let maxDepth: Int
        public let randomState: UInt64?
        public let trees: [DecisionTree.FittedState]
    }

    public func fittedState() throws -> FittedState {
        try require(!trees.isEmpty, .notFitted("RandomForest must be fitted before saving"))
        return FittedState(
            schemaVersion: fittedStateSchemaVersion,
            modelType: "RandomForest",
            nTrees: nTrees,
            maxDepth: maxDepth,
            randomState: randomState,
            trees: try trees.map { try $0.fittedState() }
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "RandomForest"
        )
        try self.init(
            nTrees: fittedState.nTrees,
            maxDepth: fittedState.maxDepth,
            randomState: fittedState.randomState
        )
        try require(!fittedState.trees.isEmpty, .notFitted("RandomForest fitted state must contain trees"))
        self.trees = try fittedState.trees.map { try DecisionTree(fittedState: $0) }
    }
}
