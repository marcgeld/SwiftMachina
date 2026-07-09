//
//  Untitled.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Pipeline Step
public enum PipelineStep {
    case transformer(any Transformer)
    case model(any Model)
}

// MARK: - Pipeline
public struct Pipeline {

    private var steps: [PipelineStep]

    public init(steps: [PipelineStep]) throws {
        try require(!steps.isEmpty, .invalidPipeline("Pipeline must contain at least one step"))
        for step in steps.dropLast() {
            if case .model = step {
                throw SwiftMachinaError.invalidPipeline("Pipeline model must be the final step")
            }
        }
        if case .model = steps.last! {
        } else {
            throw SwiftMachinaError.invalidPipeline("Pipeline must end with a model")
        }
        self.steps = steps
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) throws {

        var Xcurrent = X

        for i in 0..<steps.count {

            switch steps[i] {

            case .transformer(var t):
                try t.fit(X: Xcurrent)
                Xcurrent = try t.transform(X: Xcurrent)
                steps[i] = .transformer(t) // write back (value semantics)

            case .model(var m):
                try m.fit(X: Xcurrent, y: y)
                steps[i] = .model(m)
                return
            }
        }
    }

    // MARK: - Predict
    public func predict(X: MLXArray) throws -> MLXArray {

        var Xcurrent = X

        for step in steps {

            switch step {

            case .transformer(let t):
                Xcurrent = try t.transform(X: Xcurrent)

            case .model(let m):
                return try m.predict(X: Xcurrent)
            }
        }

        throw SwiftMachinaError.invalidPipeline("Pipeline must end with a model")
    }
}
