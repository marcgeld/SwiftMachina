//
//  Persistence.swift
//  SwiftMachina
//

import Foundation
import MLX

/// Stable, language-neutral representation of an MLX array.
///
/// The JSON form is intentionally simple:
/// `{ "shape": [rows, cols], "values": [row-major Float32 values] }`.
/// Python, CoreML host apps, and other consumers can decode this without
/// knowing anything about MLX internals.
public struct SwiftMachinaArray: Codable, Equatable {
    public let shape: [Int]
    public let values: [Float]

    public init(shape: [Int], values: [Float]) throws {
        let expectedCount = shape.reduce(1, *)
        try require(
            expectedCount == values.count,
            .invalidShape("Array payload values count must match shape product")
        )
        self.shape = shape
        self.values = values
    }

    public init(_ array: MLXArray) {
        let floatArray = array.asType(.float32)
        self.shape = floatArray.shape
        self.values = floatArray.flattened().asArray(Float.self)
    }

    public func mlxArray() throws -> MLXArray {
        let expectedCount = shape.reduce(1, *)
        try require(
            expectedCount == values.count,
            .invalidShape("Array payload values count must match shape product")
        )

        if shape.isEmpty {
            return MLXArray(values[0])
        }

        return MLXArray(values).reshaped(shape)
    }
}

/// Models conforming to this protocol can persist their complete fitted state.
///
/// The fitted-state payloads are JSON/Codable by design rather than tied to
/// MLX checkpoints, which keeps the artifacts portable across Swift, Python,
/// and CoreML-hosting apps.
public protocol FittedStatePersistable {
    associatedtype FittedState: Codable

    func fittedState() throws -> FittedState

    init(fittedState: FittedState) throws
}

public extension FittedStatePersistable {
    func saveFittedState(to url: URL, prettyPrinted: Bool = true) throws {
        let encoder = JSONEncoder()
        if prettyPrinted {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        }

        let data = try encoder.encode(try fittedState())
        try data.write(to: url, options: .atomic)
    }

    static func loadFittedState(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(FittedState.self, from: data)
        return try Self(fittedState: state)
    }
}

func requireFittedState(
    schemaVersion: Int,
    modelType: String,
    expectedModelType: String
) throws {
    try require(schemaVersion == 1, .unsupported("Unsupported fitted-state schema version"))
    try require(modelType == expectedModelType, .unsupported("Fitted-state modelType does not match \(expectedModelType)"))
}
