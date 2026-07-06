//
//  TrainTestSplit.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-29.
//

import MLX

// MARK: - Seeded PRNG (SplitMix64)

public struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
        return z ^ (z &>> 31)
    }
}

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
) -> TrainTestSplitResult {

    let n = X.shape[0]
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
            testIndices.append(contentsOf: indices.prefix(nTest))
            trainIndices.append(contentsOf: indices.dropFirst(nTest))
        }
    } else {
        var indices = Array(0..<n)
        indices.shuffle(using: &rng)
        let nTest = Int((Double(n) * testSize).rounded())
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
