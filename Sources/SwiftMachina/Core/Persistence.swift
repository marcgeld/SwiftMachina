//
//  Persistence.swift
//  SwiftMachina
//

import Foundation
import MLX

/// Schema version written to and required from every fitted-state artifact.
public let fittedStateSchemaVersion = 1

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
        try Self.validate(shape: shape, values: values)
        self.shape = shape
        self.values = values
    }

    public init(_ array: MLXArray) {
        let floatArray = array.asType(.float32)
        self.shape = floatArray.shape
        self.values = floatArray.flattened().asArray(Float.self)
    }

    // Decoded JSON is untrusted input: run the same validation the
    // designated initializer runs, which synthesized Codable would skip.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try container.decode([Int].self, forKey: .shape)
        let values = try container.decode([Float].self, forKey: .values)
        try Self.validate(shape: shape, values: values)
        self.shape = shape
        self.values = values
    }

    public func encode(to encoder: Encoder) throws {
        try require(
            values.allSatisfy(\.isFinite),
            .invalidParameter("Fitted-state arrays must contain only finite values")
        )
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(shape, forKey: .shape)
        try container.encode(values, forKey: .values)
    }

    public func mlxArray() throws -> MLXArray {
        // Shape/count consistency is an initializer invariant.
        if shape.isEmpty {
            return MLXArray(values[0])
        }
        return MLXArray(values).reshaped(shape)
    }

    private enum CodingKeys: String, CodingKey {
        case shape
        case values
    }

    private static func validate(shape: [Int], values: [Float]) throws {
        var expectedCount = 1
        for dimension in shape {
            try require(dimension >= 0, .invalidShape("Array payload dimensions must be non-negative"))
            let (product, overflow) = expectedCount.multipliedReportingOverflow(by: dimension)
            try require(!overflow, .invalidShape("Array payload shape product overflows"))
            expectedCount = product
        }
        try require(
            expectedCount == values.count,
            .invalidShape("Array payload values count must match shape product")
        )
    }
}

// MARK: - Shared Tree Serialization

/// Language-neutral tree-node payload shared by every tree-based estimator.
///
/// A node is either a leaf (only `value` set) or a complete split (all of
/// `feature`, `threshold`, `left`, and `right` set). `value` carries the
/// estimator-specific leaf payload: the majority class for `DecisionTree`,
/// the residual mean for `GradientBoosting`, the leaf weight for
/// `XGBoostClassifier`.
public final class TreeNodeState: Codable {
    public let value: Float
    public let feature: Int?
    public let threshold: Float?
    public let left: TreeNodeState?
    public let right: TreeNodeState?

    public init(
        value: Float,
        feature: Int? = nil,
        threshold: Float? = nil,
        left: TreeNodeState? = nil,
        right: TreeNodeState? = nil
    ) {
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
    }

    /// Validates the whole subtree of untrusted decoded input: every node is
    /// a leaf or a complete split, and every feature index is within
    /// `0..<nFeatures` so prediction can index feature buffers safely.
    public func validate(nFeatures: Int, model: String) throws {
        if let feature, threshold != nil, let left, let right {
            try require(
                feature >= 0 && feature < nFeatures,
                .invalidShape("\(model) node feature index must be in 0..<\(nFeatures)")
            )
            try left.validate(nFeatures: nFeatures, model: model)
            try right.validate(nFeatures: nFeatures, model: model)
            return
        }

        try require(
            feature == nil && threshold == nil && left == nil && right == nil,
            .invalidShape("\(model) node must be either a leaf or a complete split")
        )
    }
}

/// Adapter that lets each estimator's in-memory node class share the
/// `TreeNodeState` encode/decode logic instead of hand-rolling its own codec.
/// Nodes are immutable, so requiring Sendable is free and keeps the generic
/// recursion clean under strict concurrency.
protocol TreeNodeRepresentable: Sendable {
    var nodeValue: Float { get }
    var nodeFeature: Int? { get }
    var nodeThreshold: Float? { get }
    var nodeLeft: Self? { get }
    var nodeRight: Self? { get }

    init(leafValue: Float)
    init(value: Float, feature: Int, threshold: Float, left: Self, right: Self)
}

extension TreeNodeState {
    convenience init<Node: TreeNodeRepresentable>(encoding node: Node) {
        self.init(
            value: node.nodeValue,
            feature: node.nodeFeature,
            threshold: node.nodeThreshold,
            left: node.nodeLeft.map(TreeNodeState.init(encoding:)),
            right: node.nodeRight.map(TreeNodeState.init(encoding:))
        )
    }

    /// Rebuilds the in-memory node tree. Callers must run
    /// `validate(nFeatures:model:)` first; this assumes a well-formed subtree.
    func decodedNode<Node: TreeNodeRepresentable>() -> Node {
        if let feature, let threshold, let left, let right {
            return Node(
                value: value,
                feature: feature,
                threshold: threshold,
                left: left.decodedNode(),
                right: right.decodedNode()
            )
        }
        return Node(leafValue: value)
    }
}

// MARK: - Fitted-State Protocol

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
    /// Writes the fitted state as JSON. Compact by default — artifacts are
    /// machine-read; pass `prettyPrinted: true` for a human-readable file.
    /// Keys are always sorted so output is deterministic and diffable.
    func saveFittedState(to url: URL, prettyPrinted: Bool = false) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]

        let data: Data
        do {
            data = try encoder.encode(try fittedState())
        } catch let error as SwiftMachinaError {
            throw error
        } catch is EncodingError {
            throw SwiftMachinaError.invalidParameter(
                "Fitted state contains values that cannot be encoded as JSON (non-finite floats)"
            )
        }
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
    try require(
        schemaVersion == fittedStateSchemaVersion,
        .unsupported("Unsupported fitted-state schema version \(schemaVersion); expected \(fittedStateSchemaVersion)")
    )
    try require(modelType == expectedModelType, .unsupported("Fitted-state modelType does not match \(expectedModelType)"))
}
