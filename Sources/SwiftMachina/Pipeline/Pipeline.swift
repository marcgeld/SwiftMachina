//
//  Untitled.swift
//  swiftmlx
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

    public init(steps: [PipelineStep]) {
        self.steps = steps
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray) {

        var Xcurrent = X

        for i in 0..<steps.count {

            switch steps[i] {

            case .transformer(var t):
                t.fit(X: Xcurrent)
                Xcurrent = t.transform(X: Xcurrent)
                steps[i] = .transformer(t) // write back (value semantics)

            case .model(var m):
                m.fit(X: Xcurrent, y: y)
                steps[i] = .model(m)
            }
        }
    }

    // MARK: - Predict
    public func predict(X: MLXArray) -> MLXArray {

        var Xcurrent = X

        for step in steps {

            switch step {

            case .transformer(let t):
                Xcurrent = t.transform(X: Xcurrent)

            case .model(let m):
                return m.predict(X: Xcurrent)
            }
        }

        fatalError("Pipeline must end with a model")
    }
}
