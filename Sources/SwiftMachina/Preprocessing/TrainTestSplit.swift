//
//  TrainTestSplit.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-29.
//

import MLX
import SwiftNumerica

// MARK: - Seeded PRNG (SplitMix64)
// Lives upstream in SwiftNumerica since 0.1.1; the typealias keeps
// SwiftMachina's public API (and seeded sequences) unchanged.

public typealias SeededRandomNumberGenerator = SwiftNumerica.SeededRandomNumberGenerator

// MARK: - Train/Test Split Result

public struct TrainTestSplitResult {
    public let Xtrain: MLXArray
    public let Xtest: MLXArray
    public let ytrain: MLXArray
    public let ytest: MLXArray
    public let trainIndices: [Int]
    public let testIndices: [Int]
}

// MARK: - Train/Test Split
// Mirrors sklearn's train_test_split(random_state=, stratify=)

public func trainTestSplit(
    X: MLXArray,
    y: MLXArray,
    testSize: Double = 0.2,
    randomState: UInt64 = 42,
    stratify: Bool = true
) throws -> TrainTestSplitResult {
    try require(X.shape.count == 2, .invalidShape("X must be a 2D array"))
    try requireLabelVector(y, rows: X.shape[0])
    try require(testSize > 0 && testSize < 1, .invalidParameter("testSize must be in (0, 1)"))

    let n = X.shape[0]
    try require(n > 1, .invalidSplit("trainTestSplit requires at least two samples"))
    var rng = SeededRandomNumberGenerator(seed: randomState)

    let yFlat = y.reshaped([-1])

    var trainIndices: [Int]
    var testIndices: [Int]

    if stratify {
        var classBuckets: [Int: [Int]] = [:]
        for i in 0..<n {
            let label = yFlat[i].item(Int.self)
            classBuckets[label, default: []].append(i)
        }

        trainIndices = []
        testIndices = []

        for key in classBuckets.keys.sorted() {
            var indices = classBuckets[key]!
            indices.shuffle(using: &rng)

            let nTest = Int((Double(indices.count) * testSize).rounded())
            try require(nTest > 0 && nTest < indices.count, .invalidSplit("Each class must have at least one train and one test sample"))
            testIndices.append(contentsOf: indices.prefix(nTest))
            trainIndices.append(contentsOf: indices.dropFirst(nTest))
        }
    } else {
        var indices = Array(0..<n)
        indices.shuffle(using: &rng)
        let nTest = Int((Double(n) * testSize).rounded())
        try require(nTest > 0 && nTest < n, .invalidSplit("Split must produce at least one train and one test sample"))
        testIndices = Array(indices.prefix(nTest))
        trainIndices = Array(indices.dropFirst(nTest))
    }

    let trainIdx = MLXArray(trainIndices.map { Int32($0) })
    let testIdx = MLXArray(testIndices.map { Int32($0) })

    return TrainTestSplitResult(
        Xtrain: X[trainIdx],
        Xtest: X[testIdx],
        ytrain: y[trainIdx],
        ytest: y[testIdx],
        trainIndices: trainIndices,
        testIndices: testIndices
    )
}
